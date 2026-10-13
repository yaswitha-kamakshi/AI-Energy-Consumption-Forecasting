import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES_PATH = "data/processed/features.csv"
FEATURE_COLS = ["hour", "day", "month", "is_weekend", "hour_sin", "hour_cos",
                "lag_1", "lag_24", "lag_168", "roll_mean_24", "roll_std_24"]


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


if __name__ == "__main__":
    df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    X, y = df[FEATURE_COLS], df["Energy"]

    # Chronological 80/20 split (no shuffling)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Train period: {X_train.index[0]} to {X_train.index[-1]}")
    print(f"Test period:  {X_test.index[0]} to {X_test.index[-1]}\n")

    models = {
        "Naive (lag_24)": None,
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, n_jobs=-1, random_state=42),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500,
                         early_stopping=True, random_state=42)),
    }

    results, predictions = {}, {}
    for name, model in models.items():
        if model is None:
            pred = X_test["lag_24"].values       # baseline: same hour yesterday
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        predictions[name] = pred
        results[name] = evaluate(y_test, pred)
        print(f"{name:20s} MAE={results[name]['MAE']:.3f}  "
              f"RMSE={results[name]['RMSE']:.3f}  R2={results[name]['R2']:.3f}")

    # Save metrics, predictions and the best model
    metrics = pd.DataFrame(results).T.round(4)
    metrics.to_csv("outputs/metrics.csv")
    pd.DataFrame(predictions, index=y_test.index).assign(Actual=y_test.values) \
        .to_csv("outputs/test_predictions.csv")

    best = metrics.drop(index="Naive (lag_24)")["RMSE"].idxmin()
    joblib.dump(models[best], "models/best_model.pkl")
    print(f"\nBest model: {best} -> saved to models/best_model.pkl")