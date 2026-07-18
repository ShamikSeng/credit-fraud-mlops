"""Inference-input buffer + drift orchestration (Phase 3b).

This module lives in the SERVING process and deliberately imports NO Evidently. Evidently
0.7.x drags in litestar, whose `multipart` dependency conflicts irreconcilably with the
FastAPI/Starlette `python-multipart` the serving stack is pinned to (see drift_worker.py for
the full explanation). So the actual drift computation is delegated to `drift_worker.py`,
which runs as a SEPARATE PROCESS that never imports FastAPI.

What stays here (cheap, serving-safe):
- the rolling buffer of recent inference inputs + the window trigger,
- reading back the worker's JSON summary and pushing it onto the Prometheus gauges.

The worker subprocess is only ever launched from a FastAPI BackgroundTask or the manual
endpoint, so it is always off the request latency path.
"""
import collections
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import pandas as pd

from prediction_model.config import config
from prediction_model.monitoring import metrics

# maxlen == window, so once full it holds exactly the most recent window of rows; combined
# with the every-window trigger this yields non-overlapping evaluation windows.
_BUFFER = collections.deque(maxlen=config.DRIFT_WINDOW)
_LOCK = threading.Lock()
_COUNT = 0                 # total inputs recorded (drives the every-window trigger)
_LAST_SUMMARY = None       # last drift summary (served by GET /monitoring/drift)

_WORKER = os.path.join(os.path.dirname(__file__), "drift_worker.py")


def record_inference(row):
    """Append one inference input (dict of feature -> value) to the rolling buffer.

    Only the monitored features are kept (Time / extra keys ignored). Returns True when a
    drift recompute should fire: the buffer is full AND we're on a window boundary.
    """
    global _COUNT
    monitored = {f: row[f] for f in config.DRIFT_FEATURES if f in row}
    with _LOCK:
        _BUFFER.append(monitored)
        _COUNT += 1
        should_fire = (len(_BUFFER) >= config.DRIFT_WINDOW
                       and _COUNT % config.DRIFT_WINDOW == 0)
    return should_fire


def _snapshot_current():
    with _LOCK:
        if not _BUFFER:
            return None
        return pd.DataFrame(list(_BUFFER))


def _require_reference():
    if not os.path.exists(config.DRIFT_REFERENCE_FILE):
        raise FileNotFoundError(
            f"Drift reference not found at {config.DRIFT_REFERENCE_FILE}. "
            f"Generate it with: python -m prediction_model.monitoring.make_reference")


def compute_drift(current_df):
    """Compute drift for `current_df` vs the reference by invoking the isolated worker process.

    Evidently is NOT imported here — the worker (a fresh interpreter that preloads the
    standalone `multipart` and never imports FastAPI) does the computation and returns a
    compact JSON summary.
    """
    _require_reference()
    with tempfile.TemporaryDirectory() as tmp:
        current_csv = os.path.join(tmp, "current.csv")
        out_json = os.path.join(tmp, "drift.json")
        current_df.to_csv(current_csv, index=False)

        proc = subprocess.run(
            [sys.executable, _WORKER,
             "--current", current_csv,
             "--reference", config.DRIFT_REFERENCE_FILE,
             "--out", out_json,
             "--alert-share", str(config.DRIFT_ALERT_SHARE)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not os.path.exists(out_json):
            raise RuntimeError(
                f"drift worker failed (rc={proc.returncode}): {proc.stderr[-800:]}")
        with open(out_json, encoding="utf-8") as fh:
            return json.load(fh)


def run_drift_check():
    """Snapshot the buffer, compute drift vs the reference, push to the gauges, and return
    the summary. Returns None if there is nothing buffered yet."""
    global _LAST_SUMMARY
    current = _snapshot_current()
    if current is None or len(current) == 0:
        return None

    summary = compute_drift(current)
    summary["window_size"] = len(current)
    summary["timestamp"] = time.time()

    metrics.update_from_summary(summary)
    _LAST_SUMMARY = summary
    return summary


def safe_run_drift_check():
    """Background-task entry point. Monitoring must never break serving, so a drift failure
    (e.g. a missing reference or a worker error) is caught and logged, not raised into the
    server's task runner."""
    try:
        return run_drift_check()
    except Exception as exc:  # noqa: BLE001 - defensive by design
        print(f"[drift] background drift check skipped: {exc}")
        return None


def get_last_summary():
    return _LAST_SUMMARY
