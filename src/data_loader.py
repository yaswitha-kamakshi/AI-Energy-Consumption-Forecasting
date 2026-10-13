import pandas as pd

RAW_PATH = "data/raw/household_power_consumption.txt"
PROCESSED_PATH = "data/processed/hourly.csv"


def load_raw(path=RAW_PATH):
    """Load the raw UCI dataset with a proper datetime index."""
    df = pd.read_csv(path, sep=";", na_values=["?"], low_memory=False)
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.set_index("Datetime").drop(columns=["Date", "Time"])
    return df.astype(float)


if __name__ == "__main__":
    df = load_raw()
    print(df.head())
    print(df.info())
    print(df.describe())
    print("Missing values:\n", df.isna().sum())