"""A small file-based model registry (MLflow-style: versions, metrics, stages).

Layout:
    models/registry.json                 index of every model / version / stage
    models/<name>/v<N>/model.joblib      the serialised model
    models/<name>/v<N>/meta.json         params, metrics, features, timestamp
"""
import json
import time
from pathlib import Path

import joblib

ROOT = Path(__file__).parent / "models"
INDEX = ROOT / "registry.json"


def _read():
    return json.loads(INDEX.read_text()) if INDEX.exists() else {}


def _write(idx):
    ROOT.mkdir(exist_ok=True)
    INDEX.write_text(json.dumps(idx, indent=2))


def register_model(model, name, metrics, params, features):
    """Save a new version in stage 'Staging' and return the version number."""
    idx = _read()
    versions = idx.setdefault(name, {})
    version = len(versions) + 1
    vdir = ROOT / name / f"v{version}"
    vdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, vdir / "model.joblib")
    meta = {
        "name": name,
        "version": version,
        "stage": "Staging",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "params": params,
        "features": features,
    }
    (vdir / "meta.json").write_text(json.dumps(meta, indent=2))
    versions[str(version)] = {"stage": "Staging", "metrics": metrics, "created": meta["created"]}
    _write(idx)
    return version


def promote(name, version, stage="Production"):
    """Move a version to a stage; any previous holder of 'Production' is archived."""
    idx = _read()
    versions = idx[name]
    if str(version) not in versions:
        raise KeyError(f"{name} has no version {version}")
    if stage == "Production":
        for v, info in versions.items():
            if info["stage"] == "Production":
                _set_stage(name, int(v), "Archived", idx)
    _set_stage(name, int(version), stage, idx)
    _write(idx)


def _set_stage(name, version, stage, idx):
    idx[name][str(version)]["stage"] = stage
    meta_path = ROOT / name / f"v{version}" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["stage"] = stage
    meta_path.write_text(json.dumps(meta, indent=2))


def load_production(name):
    """Return (model, meta) for the model currently in stage 'Production'."""
    idx = _read()
    for v, info in idx.get(name, {}).items():
        if info["stage"] == "Production":
            vdir = ROOT / name / f"v{v}"
            return joblib.load(vdir / "model.joblib"), json.loads((vdir / "meta.json").read_text())
    raise LookupError(f"No Production version registered for '{name}'. Run train.py first.")


def list_models():
    return _read()
