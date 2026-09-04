# Runbook — Fraud Detection API

*One entry per alert. Every `runbook_url` in `observability/alert_rules.yml` links here.*

**Service:** `fraud-detection` · **Objectives:** [SLO.md](SLO.md) · **Incidents:** [POSTMORTEM-001-drift-blindspot.md](POSTMORTEM-001-drift-blindspot.md)
**Last reviewed:** 2026-09-04

---

## Progress

- [x] An entry for all 8 alert rules
- [x] Every rule annotated with a resolving `runbook_url`
- [x] Shared background section for the failure mode that spans several alerts
- [ ] Escalation contact (single maintainer today; no rotation exists)

---

## Orientation — first 60 seconds

```bash
docker compose ps                                    # what is actually running
curl -s localhost:8005/healthz                       # process alive?
curl -s -o /dev/null -w '%{http_code}\n' localhost:8005/readyz   # 200 ready / 503 no model
curl -s localhost:8005/monitoring/drift | head -c 400 # last drift summary
docker compose logs --tail=100 app                   # JSON logs; filter on "level"
```

| Surface | URL |
|---|---|
| Prometheus targets | http://localhost:9090/targets |
| Firing alerts | http://localhost:9090/alerts |
| Alertmanager | http://localhost:9093 |
| Alert notifications | `docker compose logs -f webhook-logger` |
| Grafana | http://localhost:3000 |

Logs are single-line JSON. To find drift failures specifically:

```bash
docker compose logs app | grep '"logger": "prediction_model.monitoring.drift"'
```

---

## Background — the threadpool starvation chain

**Read this before diagnosing any latency or liveness alert.** It is the failure mode that connects several of the entries below, and it is not obvious from any single metric.

1. `drift.compute_drift()` shells out to the Evidently worker with `subprocess.run(...)`.
2. Historically that call had **no `timeout=`**. A wedged worker blocked forever.
3. Starlette runs a sync `BackgroundTask` via `run_in_threadpool`, i.e. the **shared AnyIO threadpool** (default 40 slots).
4. FastAPI dispatches **every `def` path operation into that same pool.** In `main.py` that is every endpoint except `batch_predict`.
5. So N hung drift workers permanently consume N of 40 slots. At 40, *every* sync endpoint queues.
6. A sync health endpoint would queue too → the liveness probe times out → **the kubelet restarts a pod that is merely saturated**, turning a latency problem into an availability one.

One missing keyword argument; blast radius is a restart loop.

**Both mitigations are now in place** and must stay:

- `subprocess.run(..., timeout=config.DRIFT_WORKER_TIMEOUT)` — bounded, and the child is killed on timeout.
- Drift runs on a dedicated `ThreadPoolExecutor(max_workers=1)`, not on a `BackgroundTask`, so it can never contend with request handling and can never occupy more than one thread.
- `/healthz` and `/readyz` are `async def`. They run on the event loop and touch only memory. **If anyone converts them to `def`, this failure mode comes straight back** — `tests/test_health.py::test_probes_are_async` guards it.

---

## FraudAPIDown

> `up{job="fraud-detection-api"} == 0` for 1m — **critical**

**Means:** Prometheus cannot scrape the service at all. The process is gone, wedged, or the port is unreachable.

**Why it matters more than it looks:** the burn-rate alerts cannot fire on an idle service — with no traffic the error ratio is `0/0`, the series is absent, and an absent series never crosses a threshold. On a low-traffic service this is the *only* alert that catches a total outage.

**Diagnose**

```bash
docker compose ps app                       # Exited? Restarting? Up (unhealthy)?
docker compose logs --tail=200 app
curl -sv localhost:8005/healthz             # reachable at all?
docker inspect credit-fraud-mlops:latest --format '{{json .State.Health}}'
```

Check Prometheus's own view at http://localhost:9090/targets — a target `DOWN` with a connection-refused error means the process; a scrape timeout with the process up means saturation, so read the threadpool section above.

**Mitigate**

1. Container exited → `docker compose logs app` for the traceback, then `docker compose up -d app`.
2. Crash-looping on startup → almost always an import or config error, visible in the first 20 log lines. Do **not** assume a missing model: a missing model produces a *live-but-unready* pod (`/readyz` 503), never a crash loop. If it is crash-looping, something else is wrong.
3. Process up but unscrapeable → check the threadpool chain above before restarting; a restart destroys the evidence.

**Escalate:** single maintainer; no rotation.

---

## FraudAPIErrorBudgetFastBurn

> 5xx ratio > 14.4× budget over **1h AND 5m**, for 2m — **critical**

**Means:** the service is failing fast enough to exhaust a 30-day availability budget in about two days. Treat as an incident and write a postmortem.

**Diagnose**

```promql
# Which endpoint is failing?
sum by (handler) (rate(http_requests_total{status="5xx"}[5m]))

# Is it all traffic, or one route?
sum by (handler, status) (rate(http_requests_total{handler!~"/metrics|/healthz|/readyz"}[5m]))
```

```bash
docker compose logs app | grep '"level": "ERROR"' | tail -30
```

**Most likely causes, in order of how often they actually happen here**

1. **Model not loaded.** `/readyz` returns 503 and predictions 500. Check `mlruns/` is mounted and non-empty; check for a warmup failure in the logs (`model warmup failed`). In Kubernetes readiness would have pulled the pod out of service; in docker-compose there is no such gate, so failures reach clients.
2. **MLflow store unreadable or corrupted.** `_resolve_best_run_id()` runs on every request and raises `AttributeError` if the experiment is missing — remember `mlflow.get_experiment_by_name()` returns `None` rather than raising.
3. **Malformed requests reaching an unguarded path.** `POST /batch_prediction` returns HTTP 200 with an `{"error": ...}` body for a bad CSV, so it will *not* show here — but a CSV missing columns entirely can raise a pandas `KeyError` → unhandled 500.

**Mitigate**

- Model/mount problem → fix the mount and `docker compose restart app`; confirm `/readyz` returns 200 before declaring it resolved.
- Bad deploy → roll back. In Kubernetes: `kubectl rollout undo deployment/mlops`. In compose: retag the previous known-good image and `docker compose up -d app`.
- If the cause is not clear within ~15 minutes, capture `docker compose logs app > incident.log` before restarting — a restart clears the in-memory drift buffer and the model cache, destroying most of the evidence.

---

## FraudAPIErrorBudgetSlowBurn

> 5xx ratio > 6× budget over **6h AND 30m**, for 15m — **warning**

**Means:** a persistent low-level failure rate that will exhaust the month if ignored. Ticket it; do not page.

**Honest caveat:** this alert has never fired in this environment and probably cannot at demo traffic volumes — a 6h window needs sustained real traffic. It is kept because the pattern is correct under production load. See *Measurement limitations* in [SLO.md](SLO.md).

**Diagnose:** same queries as fast burn, over a longer window. Look for a *specific* failing input class rather than a systemic outage — slow burns are usually one endpoint, or one shape of request, rather than everything.

**Mitigate:** no emergency action. Open an issue, identify the failing request class, fix in the normal cycle. If the budget drops below 25%, feature work stops per the error budget policy.

---

## FraudAPILatencySLOBreach

> Under 99% of requests completing within 500ms over 1h, for 10m — **warning**

**Diagnose**

```promql
# Which handler is slow?
sum by (handler) (rate(http_request_duration_seconds_bucket{le="0.5"}[5m]))
  / sum by (handler) (rate(http_request_duration_seconds_count[5m]))

# Overview percentiles (all traffic, incl. probes — not the SLI)
histogram_quantile(0.99, sum by (le) (rate(http_request_duration_highr_seconds_bucket[5m])))
```

**Most likely causes**

1. **Cold start after a restart.** The first request pays MLflow store scan + model deserialise + SHAP `TreeExplainer` construction. The startup warmup in `main.py` exists to prevent exactly this — if it is showing up, check whether warmup failed (search logs for `model warmup failed`), because a failed warmup silently pushes the cost back onto the first real request.
2. **`_resolve_best_run_id()` on every request.** It runs a full `mlflow.search_runs` scan of the file store per prediction. As `mlruns/` grows this gets slower. The fix — pinning a registry stage — is a known deferred item.
3. **SHAP explanation cost.** `/prediction_api` explains fraud-flagged rows only, so a burst of flagged transactions raises latency for real reasons. Compare `handler="/prediction_api"` against `handler="/prediction_ui"`: if only the former is slow, this is the cause and it is expected behaviour, not a fault.
4. **Threadpool contention** — read the background section above.

**Mitigate**

- Cold start → confirm `/readyz` is 200; the condition is self-resolving once warm.
- Sustained slowness with a large `mlruns/` → prune old runs, or pin the model version.
- Only `/prediction_api` slow, with a genuine spike in fraud-flagged rows → not a defect. Note it and consider whether the SLO should exclude explanation cost.

---

## FraudDriftWorkerFailing

> `increase(fraud_drift_runs_total{result="failure"}[15m]) > 0` — **warning**

**Means:** the drift pipeline is failing. **Input-drift monitoring is blind while this is firing.**

**This is the most important entry in the runbook, because of what it protects against.** Every `fraud_input_drift_*` gauge reports *the state of the last successful run*. When a run fails, `update_from_summary()` is never reached, so the gauges keep their previous values. A completely broken drift pipeline renders on the dashboard **exactly like a healthy, non-drifting one.** Before this counter existed there was no signal at all — that is what [POSTMORTEM-001](POSTMORTEM-001-drift-blindspot.md) is about.

**So: a green drift dashboard while this alert fires is not reassurance. It is the symptom.**

**Diagnose**

```bash
docker compose logs app | grep '"logger": "prediction_model.monitoring.drift"' | tail -20
```

```promql
increase(fraud_drift_runs_total{result="failure"}[1h])
increase(fraud_drift_runs_skipped_total[1h])          # already_running vs no_data
time() - fraud_input_drift_last_run_timestamp_seconds  # how stale are the gauges really
```

Force a run and read the error directly — this is the fastest diagnostic:

```bash
curl -s -X POST localhost:8005/monitoring/drift
```

Note it returns **HTTP 200 with an error body** (`{"status": "error", "detail": ...}`), so check the body, not the status code.

**Causes, in order of likelihood**

1. **Missing or unreadable reference file.** Error text names the path and the fix. Regenerate on the host:
   ```bash
   python -m prediction_model.monitoring.make_reference
   docker compose restart app
   ```
   **Docker-specific trap:** if `reference_data.csv` does not exist on the host at first `up`, Docker creates a **directory** at the bind-mount path. `os.path.exists()` then passes and the worker fails later on `pd.read_csv` of a directory. Check with `docker compose exec app ls -la /app/prediction_model/monitoring/reference_data.csv` — if it is a directory, remove it, regenerate the file, and recreate the container.
2. **Worker timeout** (`drift worker exceeded 120.0s and was killed`). The child is killed automatically. If it recurs, the window may be too large for the host — lower `DRIFT_WINDOW` or raise `DRIFT_WORKER_TIMEOUT`, and check host CPU.
3. **Worker crash.** The last 800 characters of stderr are in the error message. A common cause is an environment change that breaks Evidently's standalone `multipart` preload.

**Mitigate**

- Fix the underlying cause, then confirm recovery: `curl -X POST localhost:8005/monitoring/drift` returns a summary and `fraud_drift_runs_total{result="success"}` increments.
- **Do not silence this alert to make the dashboard green.** Silencing it restores exactly the blind spot the postmortem was written about.

---

## FraudDriftStale

> No successful drift run in over 30 minutes, for 5m — **warning**

**Means:** backstop for failure modes the failure counter cannot see — drift never being *scheduled* (traffic stopped), or the executor thread dying.

**Expected benign cause:** low traffic. A drift run needs `DRIFT_WINDOW` (default 500) inference inputs to trigger. On an idle demo stack this alert firing is correct and uninteresting.

**Diagnose**

```promql
time() - fraud_input_drift_last_run_timestamp_seconds   # age
fraud_input_drift_window_size                           # size of last evaluated window
rate(http_requests_total{handler="/prediction_ui"}[30m]) # is there traffic at all?
```

**Mitigate**

- No traffic → generate some: `./env/Scripts/python scripts/prime_drift.py`. Not an incident.
- Traffic present but no runs → check `fraud_drift_runs_skipped_total{reason="already_running"}`. Persistent skips mean a run is stuck holding the single executor slot; restart the app and investigate the worker.

Note this alert is suppressed by an Alertmanager inhibit rule while `FraudDriftWorkerFailing` is active, since a failing worker makes staleness a guaranteed consequence rather than independent information.

---

## FraudInputDatasetDrift

> `fraud_input_drift_dataset_detected == 1` for 10m — **warning**

**Means:** the share of drifted input columns is at or above `DRIFT_ALERT_SHARE` (0.3). Live transactions no longer resemble the training distribution.

**This is a model-quality signal, not an outage.** The service is healthy; its inputs have moved. Do not restart anything.

**Diagnose**

```promql
topk(10, fraud_input_feature_drift_score)   # which features moved
fraud_input_drift_share
fraud_input_drifted_columns_count
fraud_input_drift_window_size               # is the window big enough to trust?
```

```bash
curl -s localhost:8005/monitoring/drift     # full per-feature summary
```

**First, rule out a false positive.** Confirm the drift pipeline is actually healthy (`fraud_drift_runs_total{result="success"}` recently incremented) and the window is full. A small window produces noisy shares.

**Then triage by feature importance.** Drift in features the model leans on hardest degrades PR-AUC fastest. Empirically those are **V14, V10, V4, V12** (from the SHAP attributions). Drift confined to low-influence features matters much less than the headline share suggests.

**Mitigate**

- Genuine upstream change (new payment channel, changed preprocessing) → this is a retraining trigger, not an ops fix. Open an issue.
- Reference is stale — the model was retrained but `reference_data.csv` was not regenerated → regenerate it: `python -m prediction_model.monitoring.make_reference`. **Check this before concluding the world changed;** a stale reference makes a correctly-behaving service look permanently drifted.
- Note: automated drift→retrain is deliberately *not* implemented. A retrain loop needs label latency handling, champion/challenger gating and rollback — see the scope note in `alert_rules.yml`.

---

## FraudInputDriftShareHigh

> `fraud_input_drift_share > 0.3` for 10m — **warning**

**Means:** the same condition as above, expressed against the raw share so it still fires if the dataset-flag logic changes. These two alerts fire together in normal operation.

**Known sharp edge:** `DRIFT_ALERT_SHARE` is env-overridable but the `0.3` in this rule is hard-coded. Overriding the env var silently desyncs this alert from `FraudInputDatasetDrift`. If you change one, change both.

**Diagnose and mitigate:** as [FraudInputDatasetDrift](#fraudinputdatasetdrift).

---

## Common procedures

### Roll back a bad deploy

```bash
kubectl rollout undo deployment/mlops         # Kubernetes
kubectl rollout status deployment/mlops
```

With `maxUnavailable: 0`, the replacement pod must pass its readiness probe before the old one is removed, so a rollback to a model-less image will stall rather than take the service down — which is the intended behaviour.

For docker-compose, retag the last known-good image and `docker compose up -d app`.

### Reload Prometheus rules without a restart

```bash
curl -X POST localhost:9090/-/reload
```

Enabled by `--web.enable-lifecycle`. Preserves scrape continuity — preferable to a restart mid-incident.

### Verify the whole alert path end to end

See [GAMEDAY.md](GAMEDAY.md). Injecting a bad `DRIFT_REFERENCE_FILE` is the fastest way to prove rules → Prometheus → Alertmanager → receiver is intact.

### Silence an alert

Via the Alertmanager UI at http://localhost:9093 → Silences. Always set an expiry, and record why in the comment field. An indefinite silence is how a monitoring blind spot becomes permanent.
