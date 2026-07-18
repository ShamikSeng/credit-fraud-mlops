"""Isolated drift-computation worker (Phase 3b).

Runs as its OWN process, invoked by drift.py via subprocess. It is deliberately standalone
(imports only stdlib + pandas + evidently; NOT FastAPI, NOT the prediction_model package) for
one reason: a hard, unavoidable dependency conflict.

  Why a separate process at all
  -----------------------------
  Evidently 0.7.x pulls in `litestar`, which needs the standalone `multipart` package
  (`from multipart import MultipartSegment`). FastAPI 0.103 / Starlette 0.27 need the OTHER
  `multipart` — `python-multipart` (`from multipart.multipart import parse_options_header`).
  Both claim the top-level `multipart` name and cannot coexist in one interpreter, and the
  shadow is filesystem-level (so a same-venv subprocess alone is not enough). Bumping
  python-multipart would cascade into the pinned Starlette/FastAPI serving stack.

  So the serving process keeps python-multipart and never imports evidently; THIS process
  never imports FastAPI, and — before importing evidently — pre-loads the (already installed,
  merely shadowed) standalone `multipart.py` into sys.modules so litestar/evidently import
  cleanly. Result: zero change to the pinned serving stack; evidently fully isolated.

I/O contract (pure "compare two CSVs" tool, decoupled from config/paths):
  python drift_worker.py --current <csv> --reference <csv> --out <json> --alert-share <float>
It compares the columns the two CSVs share, runs Evidently's DataDriftPreset, and writes a
compact JSON summary (the same shape metrics.update_from_summary consumes).
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import sysconfig


def _preload_standalone_multipart():
    """Make `import multipart` resolve to the standalone package litestar needs, bypassing the
    python-multipart package dir that shadows it. No-op if the standalone file isn't present."""
    standalone = os.path.join(sysconfig.get_paths()["purelib"], "multipart.py")
    if not os.path.exists(standalone):
        return  # nothing to un-shadow; let the normal import mechanism run
    spec = importlib.util.spec_from_file_location("multipart", standalone)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["multipart"] = mod


_COLUMN_RE = re.compile(r"column=([^,)]+)")


def _summarize(result_dict, monitored_features, alert_share):
    """Reduce Evidently 0.7.x `.dict()` to the compact summary the gauges consume.

    Evidently 0.7.x shape (confirmed against evidently==0.7.21):
      metrics: [
        {metric_name: 'DriftedColumnsCount(...)', config:{type:...}, value:{count, share}},
        {metric_name: 'ValueDrift(column=V1,...)', config:{type:.., column:'V1'}, value: <float>},
        ...
      ]
    ValueDrift `value` is a drift score (Wasserstein distance, normed; higher = more drift).
    """
    drifted_count = None
    drift_share = None
    per_feature = {}

    for entry in result_dict.get("metrics", []):
        name = entry.get("metric_name", "") or ""
        cfg = entry.get("config", {}) or {}
        mtype = cfg.get("type", "") or ""
        value = entry.get("value")

        if "DriftedColumnsCount" in mtype or name.startswith("DriftedColumnsCount"):
            if isinstance(value, dict):
                drifted_count = value.get("count")
                drift_share = value.get("share")
        elif "ValueDrift" in mtype or name.startswith("ValueDrift"):
            col = cfg.get("column")
            if col is None:
                m = _COLUMN_RE.search(name)
                col = m.group(1) if m else None
            if col is not None and isinstance(value, (int, float)):
                per_feature[col] = float(value)

    n_features = len(monitored_features)
    if drift_share is None and n_features:
        drift_share = (drifted_count / n_features) if drifted_count is not None else 0.0
    if drifted_count is None and drift_share is not None:
        drifted_count = round(drift_share * n_features)

    for c in monitored_features:
        per_feature.setdefault(c, 0.0)

    return {
        "dataset_drift_detected": bool((drift_share or 0.0) >= alert_share),
        "drift_share": float(drift_share or 0.0),
        "drifted_columns_count": int(drifted_count or 0),
        "n_features": n_features,
        "per_feature": per_feature,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute input drift between two CSVs.")
    parser.add_argument("--current", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--alert-share", type=float, default=0.3)
    args = parser.parse_args()

    _preload_standalone_multipart()

    import pandas as pd
    from evidently import Report
    from evidently.presets import DataDriftPreset

    current = pd.read_csv(args.current)
    reference = pd.read_csv(args.reference)
    cols = [c for c in reference.columns if c in current.columns]

    run = Report([DataDriftPreset()]).run(current_data=current[cols], reference_data=reference[cols])
    summary = _summarize(run.dict(), cols, args.alert_share)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh)
    # Also echo to stdout for easy debugging / logging.
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
