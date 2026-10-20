"""Send real test-set transactions to the running API, then run a drift check.

Usage (from the project root, with the env activated and the API running):
    python demo/send_traffic.py            # 300 normal transactions
    python demo/send_traffic.py --shift    # 300 shifted transactions (simulated drift)
    python demo/send_traffic.py --n 100 --url http://localhost:8005
"""
import argparse
import os
import sys

import requests

# allow "python demo/send_traffic.py" from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, stratified_split

SHIFTED = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14']

parser = argparse.ArgumentParser()
parser.add_argument('--shift', action='store_true', help='simulate a change in customer behaviour')
parser.add_argument('--n', type=int, default=300)
parser.add_argument('--url', default='http://localhost:8005')
args = parser.parse_args()

print('Loading test split...')
df = load_dataset(config.DATA_FILE)
_, X_test, _, _ = stratified_split(df)
start = 2000 if args.shift else 0          # use different rows for the two runs
rows = X_test.iloc[start:start + args.n].copy()
if args.shift:
    rows['Amount'] = rows['Amount'] * 3 + 50
    for c in SHIFTED:
        rows[c] = rows[c] + 1.5

try:
    requests.get(args.url + '/', timeout=5)
except requests.exceptions.ConnectionError:
    raise SystemExit(f'Cannot reach the API at {args.url}. Start it first (python main.py or docker compose up).')

frauds = 0
for i in range(len(rows)):
    r = requests.post(args.url + '/prediction_api', json=rows.iloc[i].to_dict())
    r.raise_for_status()
    frauds += r.json()['status'] == 'Fraud'
print(f'Sent {len(rows)} {"SHIFTED" if args.shift else "normal"} transactions, {frauds} predicted as fraud')

s = requests.post(args.url + '/monitoring/drift').json()
if 'drift_share' in s:
    print(f"Drift check: {s['drifted_columns_count']} of {s['n_features']} features drifted "
          f"(share = {s['drift_share']:.2f}) -> dataset drift: {s['dataset_drift_detected']}")
    top = sorted(s['per_feature'].items(), key=lambda kv: -kv[1])[:5]
    print('Top drifting features:', ', '.join(f'{k}={v:.2f}' for k, v in top))
else:
    print('Drift check response:', s)
