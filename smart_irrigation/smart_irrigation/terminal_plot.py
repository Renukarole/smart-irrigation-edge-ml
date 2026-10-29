import plotext as plt
from solar_power_regression import load, FEATURES
import joblib

df = load(None)
bundle = joblib.load("solar_model.joblib")
test = df.iloc[int(len(df) * 0.8):]
week = test.iloc[24 * 20: 24 * 27]
pred = bundle["model"].predict(week[FEATURES]).clip(0)

plt.plot(week["power_kw"].tolist(), label="Actual")
plt.plot(pred.tolist(), label="Predicted")
plt.title("One test week: actual vs predicted power (kW)")
plt.xlabel("Hour"); plt.ylabel("kW")
plt.show()