"""Send simulated UPI payments to the running UPI service, then run a drift check.

Usage (from the project root, with the env activated and the stack running):
    python demo/send_upi_traffic.py              # 500 payments from normal months (days 60-89)
    python demo/send_upi_traffic.py --festive    # 500 payments from the festive season (day 120+)
    python demo/send_upi_traffic.py --n 300 --url http://localhost:8006

Unlike the ULB demo, nothing is shifted by hand: the festive-season rows come straight from
the simulated stream, where spending, merchant mix, payees and fraud tactics really change.
"""
import argparse
import os
import sys

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction_model.upi import spec  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "prediction_model", "datasets", spec.DATA_FILE)

parser = argparse.ArgumentParser()
parser.add_argument("--festive", action="store_true", help="send festive-season payments")
parser.add_argument("--n", type=int, default=500)
parser.add_argument("--url", default="http://localhost:8006")
args = parser.parse_args()

df = pd.read_csv(DATA)
start, end = (120, spec.SIM_DAYS) if args.festive else (spec.TRAIN_END_DAY, spec.HOLDOUT_END_DAY)
rows = df[(df.txn_time >= start) & (df.txn_time < end)].sample(n=args.n, random_state=1)
rows = rows.sort_values("txn_time")

try:
    requests.get(args.url + "/", timeout=5)
except requests.exceptions.ConnectionError:
    raise SystemExit(f"Cannot reach the UPI service at {args.url}. Start it first (docker compose up).")

frauds = caught = 0
for _, r in rows.iterrows():
    body = {f: (r[f].item() if hasattr(r[f], "item") else r[f]) for f in spec.FEATURES}
    body[spec.TIME_COL] = float(r[spec.TIME_COL])
    res = requests.post(args.url + "/prediction_api", json=body)
    res.raise_for_status()
    frauds += int(r[spec.TARGET])
    caught += int(r[spec.TARGET]) and res.json()["status"] == "Fraud"
label = "FESTIVE-SEASON" if args.festive else "normal"
print(f"Sent {len(rows)} {label} payments (days {start}-{end - 1}); "
      f"{caught} of {frauds} actual frauds flagged")

s = requests.post(args.url + "/monitoring/drift").json()
if "drift_share" in s:
    print(f"Drift check: {s['drifted_columns_count']} of {s['n_features']} features drifted "
          f"(share = {s['drift_share']:.2f}) -> dataset drift: {s['dataset_drift_detected']}")
    top = sorted(s["per_feature"].items(), key=lambda kv: -kv[1])[:5]
    print("Top drifting features:", ", ".join(f"{k}={v:.2f}" for k, v in top))
else:
    print("Drift check response:", s)
