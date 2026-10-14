import sys
import joblib
import pandas as pd
from flask import Flask, request, jsonify

sys.path.insert(0, "src")
from forecast import forecast_next_n_hours

app = Flask(__name__)
model = joblib.load("models/best_model.pkl")


@app.route("/")
def home():
    return jsonify({"service": "Energy Forecast API", "usage": "POST /forecast"})


@app.route("/forecast", methods=["POST"])
def forecast():
    d = request.get_json(silent=True) or {}
    try:
        readings = [float(x) for x in d["last_168_hours_kw"]]
        start = pd.Timestamp(d["last_timestamp"])
        n = int(d.get("hours_ahead", 24))
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Provide last_timestamp, last_168_hours_kw (168 numbers) "
                                 "and optional hours_ahead"}), 400
    if len(readings) != 168 or not (1 <= n <= 168):
        return jsonify({"error": "Need exactly 168 readings and hours_ahead between 1 and 168"}), 400

    idx = pd.date_range(end=start, periods=168, freq="h")
    history = pd.Series(readings, index=idx)
    fc = forecast_next_n_hours(model, history, n)
    return jsonify({"forecast_kw": {str(k): round(v, 3) for k, v in fc.items()}})


if __name__ == "__main__":
    app.run(debug=False)