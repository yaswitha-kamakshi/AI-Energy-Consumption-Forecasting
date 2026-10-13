import pandas as pd
from data_loader import load_raw, PROCESSED_PATH


def clean_to_hourly(df):
    """Resample minute data to hourly kW and fill gaps without using future data."""
    series = df["Global_active_power"].rename("Energy")

    print(f"Missing minutes before cleaning: {series.isna().sum()}")

    # Hourly mean (kW). Hours made only of missing minutes become NaN.
    hourly = series.resample("h").mean()
    print(f"Missing hours after resampling:  {hourly.isna().sum()}")

    # Fill gaps using only past values (no leakage from the future)
    hourly = hourly.ffill()

    # Remove any duplicate timestamps (safety check)
    hourly = hourly[~hourly.index.duplicated(keep="first")]

    print(f"Missing hours after filling:     {hourly.isna().sum()}")
    return hourly.to_frame()


if __name__ == "__main__":
    raw = load_raw()
    hourly = clean_to_hourly(raw)

    hourly.to_csv(PROCESSED_PATH)
    print(f"\nSaved to {PROCESSED_PATH}")
    print(hourly.head())
    print(hourly.describe())

    # Verify the index is evenly spaced hourly
    print("\nTime gaps between rows:")
    print(hourly.index.to_series().diff().value_counts())