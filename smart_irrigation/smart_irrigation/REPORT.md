# Smart Irrigation: Real-time ML Prediction in an IoT/Edge Environment

**Domain:** Smart agriculture  **Course component:** PBA-2 project

## 1. Problem statement

Over- and under-watering both reduce crop yield and waste water. Farms typically
irrigate on fixed timers, ignoring current soil and weather conditions. This
project builds a small IoT/edge system that decides, per field zone and in real
time, whether irrigation is needed, using a machine learning model running on the
local gateway rather than in a remote cloud.

## 2. System architecture

```
Sensor nodes (simulated)  ->  Edge gateway (Flask API + ML model)  ->  Live dashboard
  4 field zones               validate -> predict -> log                 browser, polls every 2 s
  soil, temp, humidity,       model loaded from the model registry
  rain, light
```

Running inference on the gateway keeps decisions fast (about 2 ms per reading),
keeps working if the internet link drops, and avoids sending raw sensor streams
to the cloud.

| Layer | Implementation |
|---|---|
| IoT data source | `simulator.py` - four zones posting a reading each cycle over HTTP/JSON |
| Edge inference | `api.py` - Flask service, `/predict` endpoint |
| Model management | `registry.py` - versioned, staged model registry |
| Prediction output | JSON response + `logs/predictions.csv` |
| Visualisation | `static/dashboard.html` - live dashboard served by the gateway |

## 3. Dataset

Synthetic data (`generate_dataset.py`, 6000 rows, fixed seed), since real sensor
data was not required by the assignment.

| Feature | Meaning | Range in data |
|---|---|---|
| soil_moisture | % volumetric water content | 8 - 65 |
| temperature | deg C, follows a daily cycle | 11 - 43 |
| humidity | % relative humidity | 32 - 100 |
| rainfall_mm | rain in the period | 0 - 42 |
| light_lux | sunlight intensity | 0 - 91,000 |

**Target:** `irrigate` (1 = water, 0 = do not). It is derived from an agronomy-style
stress score (dry soil, heat and strong light push towards irrigating; humidity and
rain push against) plus Gaussian noise, giving a roughly balanced dataset
(53% no irrigation / 47% irrigate). The file also carries an `hour` column for
reference; it is not used as a model feature.

**Important limitation:** because the labels come from a formula, the model is
learning that formula plus noise. The accuracy below shows the pipeline works; it
is not evidence of real-world agronomic accuracy. With real field data the same
code would be retrained on logged sensor readings and operator decisions.

## 4. Model training and results

`train.py` splits the data 80/20 (stratified) and compares two models:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic regression (scaled) | 0.971 | 0.975 | 0.963 | 0.969 |
| Random forest (150 trees, depth 8) | 0.963 | 0.964 | 0.955 | 0.960 |

Logistic regression won on F1. This is expected: the underlying stress score is
close to linear, so the simpler model fits it as well as or better than the
forest. It is also tiny and fast, which suits constrained edge hardware. The
confusion matrix is in `screenshots/confusion_matrix.png`.

## 5. Model management (registry)

`registry.py` provides MLflow-style model management using only the standard
library and joblib:

- each training run registers a new **version** with its params, metrics, feature list and timestamp
- versions have a **stage**: `Staging` -> `Production` -> `Archived`
- promoting a new version to `Production` automatically archives the old one
- the edge API loads whatever is currently in `Production`, so a model update is "retrain, promote, restart the API" with no code change and a rollback path

Stored under `models/<name>/v<N>/` (`model.joblib`, `meta.json`) with an index in
`models/registry.json`. MLflow itself was not used because it could not be
installed in the environment where this was built; the registry mirrors its
concepts (`register_model`, stages, load-by-stage), so swapping in MLflow would
only change `registry.py`.

## 6. Edge API and real-time prediction

`POST /predict` accepts one reading:

1. **Validates** every field is present, numeric and within a physical range. Out-of-range values (e.g. a broken probe reporting -999) are rejected with HTTP 422 and never reach the model.
2. **Predicts** the probability of needing irrigation; probability >= 0.5 means IRRIGATE.
3. **Logs** the reading, decision, probability and latency to `logs/predictions.csv` and a rolling in-memory buffer.
4. **Returns** the action, probability, model version and latency.

Other endpoints: `/readings`, `/stats`, `/model-info`, `/health`, and `/` (dashboard).

## 7. IoT data simulation

`simulator.py` models four zones with a 30-minute simulated step per cycle:
daily temperature and light cycles, occasional rain, and soil that dries faster
when hot. It is **closed loop**: when the model returns IRRIGATE the zone's soil
moisture rises by 10-16 points on the next reading. A configurable fault rate
injects invalid sensor values to exercise input validation.

## 8. Dashboard

Served by the gateway at `http://localhost:5000` and refreshed every 2 seconds:
KPI tiles (readings, irrigate decisions, average inference time, model version and
accuracy), a status card per zone, a soil-moisture line chart per zone with
hollow markers where irrigation was triggered, and a table of the latest
predictions. See `screenshots/dashboard.png`.

## 9. Test run

One demonstration run: 4 zones x 110 cycles = 440 readings sent.

- 435 accepted and predicted, 5 rejected by validation (simulated faulty probes)
- 57 IRRIGATE decisions
- average inference latency about 1.7 ms (first request about 20 ms while warming up)
- the soil-moisture chart shows the expected sawtooth: moisture falls, the model
  triggers irrigation, moisture jumps, and the cycle repeats

## 10. Limitations and future work

- Data and labels are synthetic; retrain on real sensor logs for genuine accuracy.
- The registry is local and file-based; MLflow or a database would suit multiple users.
- The simulator uses HTTP; a real deployment would typically use MQTT from the sensor nodes.
- Possible extensions: predict water volume (regression), add soil type and crop type, drift monitoring, and automatic retraining when new data accumulates.

## 11. How to run

See `README.md`.
