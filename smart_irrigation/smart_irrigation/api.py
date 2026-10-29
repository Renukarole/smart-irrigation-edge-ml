"""Edge inference service (Flask).

Runs on the edge gateway: loads the Production model from the registry, accepts
sensor readings over HTTP, returns an irrigation decision, keeps a rolling
buffer for the dashboard and appends every prediction to logs/predictions.csv.
"""
import csv
import threading
import time
from collections import deque
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

import registry
from generate_dataset import FEATURES
from train import MODEL_NAME

app = Flask(__name__)
model, meta = registry.load_production(MODEL_NAME)

BUFFER = deque(maxlen=500)
LOCK = threading.Lock()
LOG = Path("logs/predictions.csv")
LOG.parent.mkdir(exist_ok=True)
COLUMNS = ["timestamp", "sensor_id", *FEATURES, "irrigate", "probability", "latency_ms"]
if not LOG.exists():
    LOG.write_text(",".join(COLUMNS) + "\n")

# Plausible physical ranges; anything outside is treated as a faulty sensor.
RANGES = {
    "soil_moisture": (0, 100),
    "temperature": (-10, 60),
    "humidity": (0, 100),
    "rainfall_mm": (0, 300),
    "light_lux": (0, 150000),
}


def validate(payload):
    errors = []
    for f in FEATURES:
        if f not in payload:
            errors.append(f"missing field: {f}")
            continue
        try:
            v = float(payload[f])
        except (TypeError, ValueError):
            errors.append(f"{f} must be a number")
            continue
        lo, hi = RANGES[f]
        if not lo <= v <= hi:
            errors.append(f"{f}={v} outside valid range {lo}..{hi}")
    return errors


@app.get("/")
def dashboard():
    return send_from_directory("static", "dashboard.html")


@app.get("/health")
def health():
    return jsonify(status="ok", model=meta["name"], version=meta["version"])


@app.get("/model-info")
def model_info():
    return jsonify(meta)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    errors = validate(payload)
    if errors:
        return jsonify(error="invalid reading", details=errors), 422

    start = time.perf_counter()
    row = pd.DataFrame([[float(payload[f]) for f in FEATURES]], columns=FEATURES)
    prob = float(model.predict_proba(row)[0][1])
    latency = (time.perf_counter() - start) * 1000

    result = {
        "timestamp": time.strftime("%H:%M:%S"),
        "sensor_id": str(payload.get("sensor_id", "unknown")),
        **{f: float(payload[f]) for f in FEATURES},
        "irrigate": int(prob >= 0.5),
        "probability": round(prob, 3),
        "latency_ms": round(latency, 2),
    }
    with LOCK:
        BUFFER.append(result)
        with LOG.open("a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=COLUMNS).writerow(result)
    return jsonify(action="IRRIGATE" if result["irrigate"] else "NO IRRIGATION",
                   model_version=meta["version"], **result)


@app.get("/readings")
def readings():
    limit = min(int(request.args.get("limit", 60)), 500)
    with LOCK:
        data = list(BUFFER)[-limit:]
    return jsonify(data)


@app.get("/stats")
def stats():
    with LOCK:
        data = list(BUFFER)
    n = len(data)
    return jsonify(
        total=n,
        irrigate=sum(d["irrigate"] for d in data),
        avg_latency_ms=round(sum(d["latency_ms"] for d in data) / n, 2) if n else 0,
        model_version=meta["version"],
        model_accuracy=meta["metrics"]["accuracy"],
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
