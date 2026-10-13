import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_COLS = ["hour", "day", "month", "is_weekend", "hour_sin", "hour_cos",
                "lag_1", "lag_24", "lag_168", "roll_mean_24", "roll_std_24"]


def make_row(history, ts):
    """Build one feature row for timestamp ts using only past values in history."""
    h = history  # pandas Series indexed by hourly timestamps
    last24 = h.iloc[-24:]
    return pd.DataFrame([{
        "hour": ts.hour,
        "day": ts.dayofweek,
        "month": ts.month,
        "is_weekend": int(ts.dayofweek >= 5),
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
        "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "lag_1": h.iloc[-1],
        "lag_24": h.iloc[-24],
        "lag_168": h.iloc[-168],
        "roll_mean_24": last24.mean(),
        "roll_std_24": last24.std(),
    }])[FEATURE_COLS]


def forecast_next_n_hours(model, history, n):
    """Recursive forecast: each prediction is appended and used for the next step."""
    history = history.copy()
    preds = []
    for _ in range(n):
        ts = history.index[-1] + pd.Timedelta(hours=1)
        yhat = float(model.predict(make_row(history, ts))[0])
        history.loc[ts] = yhat
        preds.append((ts, yhat))
    return pd.Series(dict(preds), name="Forecast")


if __name__ == "__main__":
    model = joblib.load("models/best_model.pkl")
    df = pd.read_csv("data/processed/hourly.csv", index_col=0, parse_dates=True)
    energy = df["Energy"]

    # Backtest: pretend "now" is the start of the last 7 days of the data
    horizon = 24 * 7
    history = energy.iloc[:-horizon]
    actual = energy.iloc[-horizon:]

    fc = forecast_next_n_hours(model, history, horizon)

    # Save forecasts
    fc.iloc[:24].to_csv("outputs/forecast_24h.csv")
    fc.to_csv("outputs/forecast_7d.csv")

    # Accuracy at different horizons
    for label, n in [("24 hours", 24), ("7 days", horizon)]:
        rmse = np.sqrt(mean_squared_error(actual.iloc[:n], fc.iloc[:n]))
        mae = mean_absolute_error(actual.iloc[:n], fc.iloc[:n])
        print(f"Backtest {label:9s} MAE={mae:.3f}  RMSE={rmse:.3f}")

    # Plot: context + actual + forecast
    context = history.iloc[-24 * 3:]
    plt.figure(figsize=(15, 5))
    plt.plot(context.index, context.values, label="History (known)", color="gray")
    plt.plot(actual.index, actual.values, label="Actual", linewidth=2)
    plt.plot(fc.index, fc.values, label="Forecast (7 days)", linestyle="--")
    plt.axvline(history.index[-1], color="red", linestyle=":", label="Forecast start")
    plt.title("7-Day Recursive Forecast vs Actual (Backtest)")
    plt.ylabel("kW"); plt.legend()
    plt.savefig("images/09_forecast_7d.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved forecasts to outputs/ and chart to images/09_forecast_7d.png")