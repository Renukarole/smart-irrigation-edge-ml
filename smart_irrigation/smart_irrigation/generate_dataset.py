"""Generate a synthetic smart-agriculture sensor dataset.

Each row is one reading from a field-zone sensor node. The target `irrigate`
(1 = water the field, 0 = do not) comes from an agronomy-style stress score
plus noise, so a model can learn it but not perfectly.
"""
import argparse
import numpy as np
import pandas as pd

FEATURES = ["soil_moisture", "temperature", "humidity", "rainfall_mm", "light_lux"]


def generate(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, n)
    # daily temperature cycle: coolest ~04:00, warmest ~15:00
    temperature = 27 + 8 * np.sin((hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 2.5, n)
    humidity = np.clip(75 - 1.2 * (temperature - 27) + rng.normal(0, 10, n), 20, 100)
    rainfall_mm = np.where(rng.random(n) < 0.18, rng.exponential(6, n), 0.0)
    daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, 1)
    light_lux = np.clip(daylight * rng.normal(60000, 9000, n), 0, 100000)
    soil_moisture = np.clip(rng.uniform(8, 60, n) + 0.8 * rainfall_mm, 5, 65)

    score = (
        (38 - soil_moisture) * 0.25
        + (temperature - 30) * 0.15
        - (humidity - 60) * 0.03
        - rainfall_mm * 0.5
        + light_lux / 100000 * 0.6
        + rng.normal(0, 0.5, n)
    )
    irrigate = (score > 0).astype(int)

    return pd.DataFrame({
        "hour": hour,
        "soil_moisture": soil_moisture.round(1),
        "temperature": temperature.round(1),
        "humidity": humidity.round(1),
        "rainfall_mm": rainfall_mm.round(1),
        "light_lux": light_lux.round(0),
        "irrigate": irrigate,
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6000)
    ap.add_argument("--out", default="data/irrigation_data.csv")
    a = ap.parse_args()
    df = generate(a.rows)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} rows to {a.out}")
    print(df["irrigate"].value_counts(normalize=True).rename("class share").round(3).to_string())
