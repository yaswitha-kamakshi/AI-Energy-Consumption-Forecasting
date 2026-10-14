import pandas as pd
import requests

df = pd.read_csv("data/processed/hourly.csv", index_col=0, parse_dates=True)
last = df["Energy"].iloc[-168:]

payload = {
    "last_timestamp": str(last.index[-1]),
    "last_168_hours_kw": last.round(3).tolist(),
    "hours_ahead": 6,
}
r = requests.post("http://127.0.0.1:5000/forecast", json=payload)
print(r.status_code)
print(r.json())