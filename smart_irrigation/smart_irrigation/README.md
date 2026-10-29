# Smart Irrigation - Real-time ML Prediction at the Edge

Simulated IoT soil/weather sensors send readings to an edge gateway. A trained
ML model on the gateway decides in real time whether each field zone needs
watering, and a live web dashboard shows the results.

```
sensor simulator --HTTP POST--> edge API (Flask + registered ML model) --> live dashboard
 (4 field zones)                 /predict  validates, predicts, logs        http://localhost:5000
```

## Run it

```bash
pip install -r requirements.txt

python generate_dataset.py          # 1. create data/irrigation_data.csv
python train.py                     # 2. train, compare, register + promote best model
python api.py                       # 3. start the edge service (leave running)

# in a second terminal
python simulator.py                 # 4. stream simulated sensor data
# open http://localhost:5000        # 5. live dashboard
```

Useful simulator options: `--interval 0.5` (faster), `--rounds 100` (stop after N),
`--fault-rate 0.1` (more broken-sensor readings, which the API rejects with HTTP 422).

## Files

| File | Purpose |
|---|---|
| `generate_dataset.py` | Synthetic sensor dataset (6000 rows) |
| `train.py` | Trains logistic regression and random forest, picks best by F1, registers it |
| `registry.py` | File-based model registry: versions, metrics, stages (Staging/Production/Archived) |
| `api.py` | Edge inference service: `/predict`, `/readings`, `/stats`, `/model-info`, `/health`, dashboard at `/` |
| `simulator.py` | IoT sensor simulator, closed loop (irrigation raises soil moisture) |
| `static/dashboard.html` | Live dashboard (plain HTML/JS, no external libraries) |
| `screenshot.py` | Captures `screenshots/dashboard.png` |
| `data/` `models/` `logs/` `screenshots/` | Dataset, registered model + `registry.json`, prediction log, images |

## API example

```bash
curl -X POST localhost:5000/predict -H 'Content-Type: application/json' \
  -d '{"sensor_id":"zone-A","soil_moisture":15,"temperature":35,"humidity":40,"rainfall_mm":0,"light_lux":70000}'
```

Returns `{"action":"IRRIGATE","probability":1.0,"model_version":1,"latency_ms":...}`.
