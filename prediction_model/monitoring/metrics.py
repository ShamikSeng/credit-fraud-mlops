"""Prometheus gauges for input-drift monitoring (Phase 3b).

Deliberately imports ONLY prometheus_client — no Evidently, no pandas-heavy work — so that:
  * importing this at app startup registers the gauges on the default registry, meaning the
    drift series appear on the existing /metrics endpoint from the first scrape (at 0 / NaN),
    before any drift run has happened; and
  * the serving process never imports Evidently just to expose metrics.

The gauges land on prometheus_client's default REGISTRY, which is exactly what
`prometheus_fastapi_instrumentator`'s `.expose(app)` serves — so no new endpoint is needed.
"""
from prometheus_client import Counter, Gauge

# --- Dataset-level gauges ---------------------------------------------------------------
DRIFT_DATASET_DETECTED = Gauge(
    "fraud_input_drift_dataset_detected",
    "1 if the last drift run flagged dataset-level input drift (share of drifted columns "
    ">= alert threshold), else 0.",
)
DRIFT_SHARE = Gauge(
    "fraud_input_drift_share",
    "Share of monitored input columns flagged as drifted in the last run (0..1).",
)
DRIFTED_COLUMNS_COUNT = Gauge(
    "fraud_input_drifted_columns_count",
    "Number of monitored input columns flagged as drifted in the last run.",
)
DRIFT_WINDOW_SIZE = Gauge(
    "fraud_input_drift_window_size",
    "Number of buffered inference rows compared against the reference in the last run.",
)
DRIFT_LAST_RUN_TS = Gauge(
    "fraud_input_drift_last_run_timestamp_seconds",
    "Unix timestamp of the last completed drift run.",
)

# --- Per-feature gauge ------------------------------------------------------------------
# One time series per monitored feature (V1..V28 + Amount = 29 series; low cardinality).
# Value is Evidently's per-column drift score for that feature in the last run.
FEATURE_DRIFT_SCORE = Gauge(
    "fraud_input_feature_drift_score",
    "Per-feature drift score from the last run (statistic/distance; higher = more drift).",
    ["feature"],
)


# --- Run-outcome counters (Phase 4 SRE) -------------------------------------------------
# The gauges above are all "state of the last SUCCESSFUL run". That leaves a blind spot
# they cannot express: when a run FAILS, update_from_summary is never reached, so every
# gauge keeps its previous value and a permanently broken drift pipeline is
# indistinguishable on a dashboard from a healthy, non-drifting one. These counters are
# the signal for "is the monitoring itself working?" — monitoring the monitoring.
#
# Registered at module import (never inside a function) so a re-import cannot raise
# Duplicated timeseries against the global REGISTRY that tests assert on.
DRIFT_RUNS = Counter(
    "fraud_drift_runs_total",
    "Drift-check runs by outcome. `failure` includes worker crashes, a missing reference "
    "file, and worker timeouts.",
    ["result"],
)
DRIFT_RUNS_SKIPPED = Counter(
    "fraud_drift_runs_skipped_total",
    "Drift checks that were requested but not executed, by reason.",
    ["reason"],
)

# Pre-initialise every label value so the series exist at 0 from the first scrape. Without
# this, `increase(fraud_drift_runs_total{result="failure"}[15m]) > 0` cannot fire on the
# very first failure: the series would not exist yet, and there is nothing to increase
# from. Same reasoning as the module docstring's note about the gauges appearing at 0.
for _result in ("success", "failure"):
    DRIFT_RUNS.labels(result=_result)
for _reason in ("already_running", "no_data"):
    DRIFT_RUNS_SKIPPED.labels(reason=_reason)


def update_from_summary(summary):
    """Push a drift summary (the dict produced by drift.compute_drift) onto the gauges.

    Every monitored feature is set each run — including the non-drifted ones (score, or 0.0
    if absent) — so a feature that stops drifting visibly falls back on the dashboard instead
    of holding a stale high value.
    """
    DRIFT_DATASET_DETECTED.set(1 if summary.get("dataset_drift_detected") else 0)
    DRIFT_SHARE.set(float(summary.get("drift_share", 0.0)))
    DRIFTED_COLUMNS_COUNT.set(int(summary.get("drifted_columns_count", 0)))
    DRIFT_WINDOW_SIZE.set(int(summary.get("window_size", 0)))
    DRIFT_LAST_RUN_TS.set(float(summary.get("timestamp", 0.0)))

    per_feature = summary.get("per_feature", {}) or {}
    for feature, score in per_feature.items():
        FEATURE_DRIFT_SCORE.labels(feature=feature).set(float(score))
