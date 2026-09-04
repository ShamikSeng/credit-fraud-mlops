"""Inference-input buffer + drift orchestration (Phase 3b).

This module lives in the SERVING process and deliberately imports NO Evidently. Evidently
0.7.x drags in litestar, whose `multipart` dependency conflicts irreconcilably with the
FastAPI/Starlette `python-multipart` the serving stack is pinned to (see drift_worker.py for
the full explanation). So the actual drift computation is delegated to `drift_worker.py`,
which runs as a SEPARATE PROCESS that never imports FastAPI.

What stays here (cheap, serving-safe):
- the rolling buffer of recent inference inputs + the window trigger,
- reading back the worker's JSON summary and pushing it onto the Prometheus gauges.

The worker subprocess is launched from a dedicated single-thread executor (see _EXECUTOR
below) or the manual endpoint, so it is genuinely off the request latency path — a
BackgroundTask would NOT be, because Starlette awaits those inside the timed response.
"""
import collections
import concurrent.futures
import json
import logging
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

logger = logging.getLogger(__name__)

# Drift runs execute HERE, on a dedicated single-thread executor, and not as a Starlette
# BackgroundTask. Two reasons, both learned the hard way (POSTMORTEM-001):
#
# 1. LATENCY ACCOUNTING. Starlette awaits BackgroundTask inside the response
#    (starlette/responses.py: `await self.background()` after the body is sent), which is
#    still inside the instrumentator's timing block. Every 500th request therefore recorded
#    serving time PLUS the full Evidently subprocess wall time — latency the client never
#    experienced. On a 99.5%-under-500ms objective that one-in-500 phantom burns a large
#    fraction of the error budget for nothing.
#
# 2. THREADPOOL ISOLATION. A sync BackgroundTask runs via run_in_threadpool, i.e. the
#    shared AnyIO pool (default 40 slots) that also serves every `def` endpoint. Slow or
#    wedged drift runs there compete directly with request handling. max_workers=1 caps the
#    blast radius at exactly one thread and gives us "never two drift runs at once" for free.
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="drift-worker")
_RUNNING = threading.Event()


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

        try:
            proc = subprocess.run(
                [sys.executable, _WORKER,
                 "--current", current_csv,
                 "--reference", config.DRIFT_REFERENCE_FILE,
                 "--out", out_json,
                 "--alert-share", str(config.DRIFT_ALERT_SHARE)],
                capture_output=True, text=True,
                # Without this the call blocks forever on a wedged worker: the executor
                # thread is consumed permanently, every later drift run is skipped, and the
                # gauges hold their last good values. Nothing in the stack reports it.
                # subprocess.run kills the child on timeout before re-raising.
                timeout=config.DRIFT_WORKER_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"drift worker exceeded {config.DRIFT_WORKER_TIMEOUT}s and was killed"
            ) from exc

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

    # Outcome accounting lives HERE, not in safe_run_drift_check, so that every path that
    # runs a drift check is counted: the scheduled executor path AND the manual
    # POST /monitoring/drift endpoint. Counting in one caller would leave the manual path
    # invisible, which is precisely the kind of partial instrumentation that produces a
    # dashboard you cannot trust.
    try:
        summary = compute_drift(current)
    except Exception:
        metrics.DRIFT_RUNS.labels(result="failure").inc()
        raise

    summary["window_size"] = len(current)
    summary["timestamp"] = time.time()

    metrics.update_from_summary(summary)
    metrics.DRIFT_RUNS.labels(result="success").inc()
    _LAST_SUMMARY = summary
    return summary


def safe_run_drift_check():
    """Worker entry point. Monitoring must never break serving, so a drift failure (a
    missing reference, a worker crash, a timeout) is counted and logged, never raised.

    Swallowing the exception is still right — but on its own it was how a broken drift
    pipeline stayed invisible for hours. The counter below is what makes the failure
    observable, because the gauges cannot express it: they simply keep their last values.
    """
    try:
        summary = run_drift_check()
    except Exception as exc:  # noqa: BLE001 - defensive by design; see docstring
        # The failure counter is incremented inside run_drift_check so the manual endpoint
        # is counted too; here we only add the log line and stop the exception.
        logger.error("drift check failed: %s", exc, exc_info=True)
        return None

    if summary is None:
        metrics.DRIFT_RUNS_SKIPPED.labels(reason="no_data").inc()
        return None

    logger.info(
        "drift check complete: share=%.3f drifted_columns=%s window=%s",
        summary.get("drift_share", 0.0),
        summary.get("drifted_columns_count"),
        summary.get("window_size"),
    )
    return summary


def _run_and_clear():
    try:
        return safe_run_drift_check()
    finally:
        _RUNNING.clear()


def schedule_drift_check():
    """Queue a drift run on the dedicated executor and return immediately.

    Returns True if a run was queued, False if one was already in flight. Drift runs are
    idempotent over a rolling buffer, so dropping an overlapping request is strictly better
    than letting slow runs pile up behind each other and outlive their own window.
    """
    if _RUNNING.is_set():
        metrics.DRIFT_RUNS_SKIPPED.labels(reason="already_running").inc()
        logger.warning("drift check skipped: previous run still in flight")
        return False
    _RUNNING.set()
    try:
        _EXECUTOR.submit(_run_and_clear)
    except Exception:  # noqa: BLE001 - e.g. interpreter shutdown
        _RUNNING.clear()
        raise
    return True


def get_last_summary():
    return _LAST_SUMMARY
