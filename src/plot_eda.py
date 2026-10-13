import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/hourly.csv", index_col=0, parse_dates=True)

# Full time series
plt.figure(figsize=(15, 5))
df["Energy"].plot(title="Hourly Household Energy Consumption (kW)")
plt.ylabel("kW")
plt.savefig("images/02_consumption_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()

# Average by hour of day (human-behavior pattern)
df.groupby(df.index.hour)["Energy"].mean().plot(
    kind="bar", figsize=(10, 4), title="Average Consumption by Hour of Day"
)
plt.xlabel("Hour")
plt.ylabel("kW")
plt.savefig("images/03_hourly_pattern.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved plots to images/")