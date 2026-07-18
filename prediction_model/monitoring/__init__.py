"""Phase 3b: input-drift monitoring.

Split so the serving process NEVER imports Evidently (Evidently 0.7.x pulls litestar, whose
`multipart` dependency conflicts irreconcilably with the FastAPI/Starlette `python-multipart`
the serving stack is pinned to — see drift_worker.py):

- `metrics`      : Prometheus gauge definitions ONLY (prometheus_client, no Evidently). Imported
                   at app startup so the drift gauges appear on /metrics immediately.
- `drift`        : the rolling inference-input buffer + orchestration. Imports NO Evidently;
                   it delegates the actual computation to a separate worker process.
- `drift_worker` : a standalone script (its own interpreter) that preloads the standalone
                   `multipart`, imports Evidently, computes drift between two CSVs, and writes
                   a compact JSON summary. Launched only from a BackgroundTask / manual endpoint.
- `make_reference` : standalone script that regenerates the training-distribution reference.
"""
