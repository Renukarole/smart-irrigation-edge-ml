"""Simulated IoT sensor network: four field zones posting readings to the edge API.

Closed loop: when the model says IRRIGATE, that zone's soil moisture rises on the
next reading, so the dashboard shows the system reacting to its own decisions.
"""
import argparse
import math
import random
import time

import requests

ZONES = ["zone-A", "zone-B", "zone-C", "zone-D"]


class Zone:
    def __init__(self, name, rng):
        self.name = name
        self.rng = rng
        self.soil = rng.uniform(25, 55)

    def reading(self, tick, faulty=False):
        hour = (6 + tick * 0.5) % 24            # each tick = 30 simulated minutes
        temp = 27 + 8 * math.sin((hour - 9) / 24 * 2 * math.pi) + self.rng.gauss(0, 1.2)
        hum = max(20, min(100, 75 - 1.2 * (temp - 27) + self.rng.gauss(0, 4)))
        rain = self.rng.expovariate(1 / 6) if self.rng.random() < 0.06 else 0.0
        daylight = max(0, math.sin((hour - 6) / 12 * math.pi))
        lux = max(0, daylight * self.rng.gauss(60000, 5000))
        # soil dries faster when hot and sunny, and gains from rain
        self.soil = max(5, min(65, self.soil - (1.2 + 0.12 * max(temp - 20, 0)) + 0.8 * rain))
        r = {
            "sensor_id": self.name,
            "soil_moisture": round(self.soil, 1),
            "temperature": round(temp, 1),
            "humidity": round(hum, 1),
            "rainfall_mm": round(rain, 1),
            "light_lux": round(lux),
        }
        if faulty:                                # simulate a broken probe
            r["soil_moisture"] = -999
        return r

    def irrigate(self):
        self.soil = min(65, self.soil + self.rng.uniform(10, 16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5000/predict")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between rounds")
    ap.add_argument("--rounds", type=int, default=0, help="0 = run until Ctrl+C")
    ap.add_argument("--fault-rate", type=float, default=0.02, help="chance a reading is a bad sensor value")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    zones = [Zone(n, rng) for n in ZONES]
    tick = 0
    print(f"Sending to {a.url} every {a.interval}s (Ctrl+C to stop)")
    try:
        while a.rounds == 0 or tick < a.rounds:
            for z in zones:
                payload = z.reading(tick, faulty=rng.random() < a.fault_rate)
                try:
                    resp = requests.post(a.url, json=payload, timeout=3)
                except requests.RequestException as e:
                    print(f"  ! cannot reach API: {e}")
                    continue
                if resp.status_code == 200:
                    out = resp.json()
                    print(f"{z.name} moisture={payload['soil_moisture']:5.1f} temp={payload['temperature']:4.1f} "
                          f"-> {out['action']:<14} p={out['probability']:.2f} ({out['latency_ms']} ms)")
                    if out["irrigate"]:
                        z.irrigate()
                else:
                    print(f"{z.name} rejected ({resp.status_code}): {resp.json().get('details')}")
            tick += 1
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("\nSimulator stopped.")


if __name__ == "__main__":
    main()
