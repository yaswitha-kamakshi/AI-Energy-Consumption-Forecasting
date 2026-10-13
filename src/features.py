import numpy as np
import pandas as pd

PROCESSED_PATH = "data/processed/hourly.csv"
FEATURES_PATH = "data/processed/features.csv"


def build_features(df):
    """Create calendar, cyclical, lag and rolling features from hourly Energy."""
    out = df.copy()

    # --- Calendar features ---
    out["hour"] = out.index.hour
    out["day"] = out.index.dayofweek          # 0 = Monday
    out["month"] = out.index.month
    out["is_weekend"] = (out["day"] >= 5).astype(int)

    # --- Cyclical encoding (so hour 23 is "close" to hour 0) ---
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    # --- Lag features (past usage) ---
    out["lag_1"] = out["Energy"].shift(1)       # 1 hour ago
    out["lag_24"] = out["Energy"].shift(24)     # same hour yesterday
    out["lag_168"] = out["Energy"].shift(168)   # same hour last week

    # --- Rolling features (shift first so the current row is NOT included) ---
    out["roll_mean_24"] = out["Energy"].shift(1).rolling(24).mean()
    out["roll_std_24"] = out["Energy"].shift(1).rolling(24).std()

    # Drop the first rows that have NaN because of the lags
    out = out.dropna()
    return out


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_PATH, index_col=0, parse_dates=True)
    feats = build_features(df)

    feats.to_csv(FEATURES_PATH)
    print(f"Rows before: {len(df)}  |  after: {len(feats)}")
    print(f"Saved to {FEATURES_PATH}\n")
    print(feats.head())
    print("\nMissing values:", feats.isna().sum().sum())

    # Leakage check: lag_24 must equal the value 24 hours earlier
    t = feats.index[100]
    same = np.isclose(feats.loc[t, "lag_24"], df.loc[t - pd.Timedelta(hours=24), "Energy"])
    print("lag_24 check passed:", same)

    # Correlation with the target
    print("\nCorrelation with Energy:")
    print(feats.corr()["Energy"].sort_values(ascending=False).round(3))