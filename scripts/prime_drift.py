"""
Primes the running prediction service with 500 sampled non-fraud rows,
then triggers a drift check.

Paths are relative to the repo root, so run it from there:
    ./env/Scripts/python scripts/prime_drift.py
"""

import sys
import pandas as pd
import requests

DATA_PATH = "prediction_model/datasets/creditcard.csv"
PREDICTION_URL = "http://localhost:8005/prediction_ui"
DRIFT_URL = "http://localhost:8005/monitoring/drift"
SAMPLE_SIZE = 500
RANDOM_STATE = 42
TIMEOUT = 10


def main():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: could not find {DATA_PATH} — run this from the project root.")
        sys.exit(1)

    if "Class" not in df.columns:
        print("ERROR: expected a 'Class' column in the dataset, none found.")
        sys.exit(1)

    non_fraud = df[df["Class"] == 0].drop(columns=["Class"])
    if len(non_fraud) < SAMPLE_SIZE:
        print(f"WARNING: only {len(non_fraud)} non-fraud rows available, "
              f"requested {SAMPLE_SIZE}. Sampling all available rows instead.")
        sample = non_fraud
    else:
        sample = non_fraud.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    success, failed = 0, 0
    for idx, row in sample.iterrows():
        try:
            resp = requests.post(PREDICTION_URL, json=row.to_dict(), timeout=TIMEOUT)
            if resp.status_code == 200:
                success += 1
            else:
                failed += 1
                print(f"Row {idx}: HTTP {resp.status_code} — {resp.text[:200]}")
        except requests.exceptions.RequestException as e:
            failed += 1
            print(f"Row {idx}: request failed — {e}")

    print(f"\nPrimed {success}/{len(sample)} rows ({failed} failed).")

    if success == 0:
        print("No rows succeeded — skipping drift check. Is the service running on :8005?")
        sys.exit(1)

    try:
        drift_resp = requests.post(DRIFT_URL, timeout=TIMEOUT)
        print(f"Drift check: HTTP {drift_resp.status_code}")
        print(drift_resp.text[:500])
    except requests.exceptions.RequestException as e:
        print(f"Drift check request failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
