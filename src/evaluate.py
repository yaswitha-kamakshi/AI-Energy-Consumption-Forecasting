import pandas as pd
import matplotlib.pyplot as plt
import joblib

FEATURE_COLS = ["hour", "day", "month", "is_weekend", "hour_sin", "hour_cos",
                "lag_1", "lag_24", "lag_168", "roll_mean_24", "roll_std_24"]

preds = pd.read_csv("outputs/test_predictions.csv", index_col=0, parse_dates=True)
metrics = pd.read_csv("outputs/metrics.csv", index_col=0)

# 1. Actual vs predicted (first 7 days of the test period)
n = 24 * 7
plt.figure(figsize=(15, 5))
plt.plot(preds.index[:n], preds["Actual"][:n], label="Actual", linewidth=2)
plt.plot(preds.index[:n], preds["Random Forest"][:n], label="Random Forest", linestyle="--")
plt.title("Actual vs Predicted Energy Consumption (7 days of test data)")
plt.ylabel("kW"); plt.legend()
plt.savefig("images/05_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Model comparison (RMSE)
metrics["RMSE"].plot(kind="bar", figsize=(9, 5), color="steelblue",
                     title="Model Comparison (lower RMSE is better)")
plt.ylabel("RMSE (kW)"); plt.xticks(rotation=20)
plt.savefig("images/06_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Error distribution
errors = preds["Actual"] - preds["Random Forest"]
plt.figure(figsize=(9, 5))
plt.hist(errors, bins=60, color="darkorange", edgecolor="black")
plt.title("Prediction Error Distribution (Random Forest)")
plt.xlabel("Actual - Predicted (kW)"); plt.ylabel("Count")
plt.savefig("images/07_error_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# 4. Feature importance
model = joblib.load("models/best_model.pkl")
imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
imp.plot(kind="barh", figsize=(9, 6), title="Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.savefig("images/08_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved 4 charts to images/")
print(f"Mean error: {errors.mean():.3f} kW | Std: {errors.std():.3f} kW")
