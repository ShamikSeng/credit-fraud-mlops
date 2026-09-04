# Service Level Objectives — Fraud Detection API

*What "working" means for this service, how it is measured, and what we do when the measurement says it is not.*

**Service:** `fraud-detection` (FastAPI, single uvicorn process)
**Prometheus job:** `fraud-detection-api`
**Dashboards:** Grafana → *Fraud API — SLOs & Pipeline Health*, *Fraud Model — Input Drift*
**Runbook:** [RUNBOOK.md](RUNBOOK.md)
**Owner:** single maintainer; no on-call rotation (see *Error budget policy*)
**Last reviewed:** 2026-09-04

---

## Progress

- [x] SLIs defined against metrics that actually exist in this stack
- [x] Availability and latency objectives set, with error budgets
- [x] Recording rules for every SLI window (`observability/recording_rules.yml`)
- [x] Multiwindow burn-rate alerts wired to Alertmanager
- [x] Measurement limitations documented rather than hidden
- [x] Load-test numbers measured, and the 180x latency defect they exposed fixed

---

## Why these SLIs

The service has exactly one job a user cares about: **return a fraud verdict, correctly and quickly, when a transaction is scored.** Two things can break that from the user's side — it errors, or it is too slow. So there are two SLIs, and deliberately not more. Every additional SLI is another thing that can page someone, and a page nobody acts on trains people to ignore pages.

CPU, memory and request rate are all monitored, but none of them are SLIs. They are *causes*, not *symptoms*. A saturated CPU that still serves every request under 500ms is not an incident.

### SLI 1 — Availability

> The proportion of requests that do not fail with a server error.

```promql
sum by (job) (rate(http_requests_total{status!="5xx",handler!~"/metrics|/healthz|/readyz"}[5m]))
/
sum by (job) (rate(http_requests_total{handler!~"/metrics|/healthz|/readyz"}[5m]))
```

**4xx counts as GOOD, and that is a real decision, not an oversight.** A `422` from the pydantic `TransactionFeatures` schema is the service working exactly as designed: it received a malformed transaction and correctly refused it. If 4xx counted against availability, a client shipping bad payloads could exhaust our error budget and page us about their bug.

**Probe and scrape traffic is excluded, and this is load-bearing.** Kubernetes probes hit `/healthz` and `/readyz` every 10s; Prometheus scrapes `/metrics` every 15s. That is roughly ten guaranteed-fast, guaranteed-`2xx` requests per minute with no real user behind any of them. At demo traffic volumes they dominate the ratio and pin the SLI at `1.0` — an SLI that cannot go down is decoration. The `handler!~` filter appears on every rule in `recording_rules.yml` for this reason.

### SLI 2 — Latency

> The proportion of requests that complete in under 500ms.

```promql
sum by (job) (rate(http_request_duration_seconds_bucket{le="0.5",handler!~"/metrics|/healthz|/readyz"}[5m]))
/
sum by (job) (rate(http_request_duration_seconds_count{handler!~"/metrics|/healthz|/readyz"}[5m]))
```

**A threshold ratio, not a percentile.** `prometheus_fastapi_instrumentator` 6.1.0 gives the per-handler histogram only four buckets — `le` = 0.1, 0.5, 1, `+Inf`. `histogram_quantile(0.99, ...)` over those returns `+Inf` for any p99 above one second, i.e. it goes blind exactly when latency becomes a problem. "What fraction of requests beat 500ms" is answerable precisely with these buckets, and it is the formulation the Google SRE Workbook prefers anyway: it maps directly onto "how many users had a bad experience".

`_count` is used as the denominator rather than `le="+Inf"`. They are numerically identical, but `_count` survives someone changing the bucket boundaries later.

---

## Objectives and error budgets

| SLI | Objective (30-day rolling) | Error budget | Budget in a 30-day month |
|---|---|---|---|
| Availability | **99.5%** of requests non-5xx | 0.5% | ~3h 39m of total failure |
| Latency | **99%** of requests under 500ms | 1% | ~7h 18m over threshold |

**Why 99.5% and not 99.9%.** Three nines on a single-replica service with no redundancy, no autoscaling and a model that must warm on start would be a number invented to look impressive. `replicas: 1` means every deliberate rollout is a brief outage, and the budget has to be able to absorb the deploys we actually intend to do. 99.5% is a target this architecture can genuinely meet; when the Known Gap below is closed and the service can run multiple replicas, tightening it is the right follow-up.

**Why 500ms.** Fraud scoring sits inline in a payment authorisation path, where the whole authorisation typically budgets ~1s end to end. 500ms leaves room for the network and the caller's own work. Measured steady-state latency is far below this — the budget exists for the tail, not the median.

### Error budget policy

There is no rotation to page, so the policy is about *what changes*, not *who wakes up*:

- **Budget healthy (>25% remaining):** ship normally.
- **Budget below 25%:** no new features until the burn is understood. Reliability work takes priority.
- **Budget exhausted:** feature work stops. The only changes that ship are ones that reduce the burn, plus the follow-up actions from the relevant postmortem.
- **Fast-burn alert fires:** treat as an incident, write a postmortem (see [POSTMORTEM-001-drift-blindspot.md](POSTMORTEM-001-drift-blindspot.md) for the format).

### Burn-rate alerting

Multiwindow, multi-burn-rate, per the SRE Workbook:

| Alert | Long window | Short window | Burn rate | Budget consumed in | Response |
|---|---|---|---|---|---|
| `FraudAPIErrorBudgetFastBurn` | 1h | 5m | 14.4x | ~2 days | Page — treat as incident |
| `FraudAPIErrorBudgetSlowBurn` | 6h | 30m | 6x | ~5 days | Ticket — investigate in hours |

The short window is ANDed with the long one so the alert *clears* promptly once the burn stops, instead of smouldering for a full hour after the problem is fixed. That pairing is the entire point of the pattern.

---

## Measurement limitations

Stated plainly, because an SLO whose caveats live only in the author's head is not a shared contract.

**1. The latency histogram has no `status` label.** `http_request_duration_seconds` is labelled `method` and `handler` only. Slow *failed* requests are counted in the latency SLI alongside slow successful ones. In practice a 5xx here is usually fast (an exception, not a timeout), so this inflates the latency SLI slightly rather than deflating it — but it is measured, not assumed away.

**2. Zero traffic produces no signal, by design.** With no requests the ratio is `0/0`, the series is absent, and an absent series never crosses a threshold. This is correct — no traffic means no users having a bad time, and no budget being burned — but it means **the burn-rate alerts cannot detect an outage on an idle service.** `FraudAPIDown` (`up == 0`) is the alert that covers that case, and it is why that rule exists separately.

**3. The slow-burn alert has never fired here, and probably cannot at demo volumes.** A 6h window needs sustained real traffic to accumulate a meaningful ratio. It is kept because the pattern is correct and it would work under production load — not because it has been demonstrated. Only the fast-burn path has been exercised end to end.

**4. `http_request_duration_highr_seconds` cannot be filtered.** It has fine buckets (0.01s → 60s) but *no labels at all*, so probe and scrape traffic cannot be excluded from it. The percentile panel on the SLO dashboard uses it for overview only and is explicitly labelled "all traffic". It is not the SLI.

**5. SLIs are measured server-side.** Everything here is recorded inside the application process, so it excludes network time, load-balancer time, and anything that prevents a request from arriving at all. Real user-facing availability is therefore somewhat *worse* than these numbers. Closing that gap needs black-box probing from outside the cluster — noted as future work, not claimed.

---

## Known Gap — this service cannot safely run more than one replica

`deployment.yml` pins `replicas: 1`. That is a constraint discovered while doing this work, not a default nobody revisited, and it is the most interesting limitation in the system.

**What actually breaks.** `prediction_model/monitoring/drift.py` keeps `_BUFFER` (a 500-slot `deque`) and `_COUNT` as **per-process globals**. With two replicas:

- Each pod fills its own buffer at roughly half the request rate, so each drift evaluation runs over a biased half-sample. **The drift statistic itself becomes wrong**, not merely duplicated.
- `observability/prometheus.yml` scrapes a single static target (`app:8005`). Two pods behind one Service do not produce two series distinguishable by `instance` — they produce **one series that alternates between two pods' unrelated drift states** on each scrape. That artifact is considerably worse than the single-replica limitation it would be hiding.

**What does *not* break, which is worth separating.** `predict._CACHE` also duplicates per replica, but the consequence is only 2× memory and one extra cold start. It is wasteful, not incorrect. Knowing which shared state is a correctness problem and which is merely cost is the difference between "it doesn't scale" and understanding why.

**What closing it requires**, roughly in order:

1. Externalise the buffer — Redis `LPUSH`/`LTRIM` for the rolling window, plus a lock so exactly one pod runs the drift evaluation, **or** split drift into its own single-replica Deployment (`strategy: Recreate`) consuming from that shared store.
2. Move the gauges to whichever process owns the computation (or a Pushgateway), so there is one authoritative series.
3. Switch Prometheus from `static_configs` to `kubernetes_sd_configs` with `role: pod`, so each pod is scraped as its own target with a real `instance` label.
4. *Then* raise `replicas`, add a `PodDisruptionBudget` (`maxUnavailable: 1` — at `replicas: 1` a `minAvailable: 1` PDB blocks node drains entirely), and an HPA on `resources.requests.cpu`.

Until then the honest position is: this service is available to the extent that one pod is available, and the SLO is set accordingly.

---

## Measured capacity

Measured 2026-09-04 on the local `docker compose` stack (Docker Desktop, Windows 11), one
uvicorn process, `DRIFT_ENABLED=false` so the numbers are serving latency and not
monitoring overhead. Load generated by `loadtest/k6_prediction.js` against
`POST /prediction_ui`.

### The finding that mattered

The first load test **failed badly**: 1.08 req/s, median 8.5s, p95 30.2s — with **zero
errors**. Nothing was broken; the service was simply that slow, and no amount of
concurrency helped.

Profiling inside the container found it in one measurement:

| Operation | Cost per request |
|---|---|
| `_resolve_best_run_id()` — `mlflow.search_runs` over the run store | **3053 ms** |
| `model.predict()` — the actual work | **17 ms** |

**A 180x overhead, paid on every single prediction.** About 99.4% of request latency was
MLflow re-reading run metadata to answer a question whose answer had not changed. The
per-request run resolution was a known deferred item noted in `predict.py`'s own comments;
the load test is what turned "known suboptimal" into "this makes the SLO unreachable".

Fixed by caching the resolution with a TTL (`MODEL_RESOLVE_TTL`, default 300s), holding
the lock across the lookup so a cache miss under concurrency cannot stampede.

| | Before | After |
|---|---|---|
| Sequential latency | ~3000 ms | **~50 ms** |
| Throughput (conformance profile) | 1.08 req/s | **37.3 req/s** |
| p95 latency | 30.2 s | **82.7 ms** |

The tradeoff is explicit: a newly trained better run is picked up within `MODEL_RESOLVE_TTL`
rather than on the very next request. For a model retrained daily or slower, five minutes
of staleness is irrelevant; three seconds per prediction is not.

### Conformance run (the SLO gate)

`docker compose --profile loadtest run --rm k6` — 2 VUs, 85s. **All thresholds pass.**

| Metric | Value |
|---|---|
| Throughput | 37.3 req/s |
| Latency avg / med | 47.0 ms / 43.6 ms |
| p90 / p95 | 69.0 ms / 82.7 ms |
| 5xx rate | 0.00% (0 of 3170) |
| Checks passed | 100% (6338) |

This profile deliberately runs *within* measured capacity, so a failure means something
regressed rather than that the load was unreasonable. A load test that always fails trains
people to ignore it.

### Capacity curve (25s per point, concurrency swept manually)

| Concurrent requests | Throughput | Median | p95 | Meets 500ms objective |
|---|---|---|---|---|
| 1 | 17.9 req/s | 44.7 ms | 132.8 ms | yes |
| 2 | 29.5 req/s | 58.9 ms | 128.6 ms | yes |
| 4 | 15.8 req/s | 232.5 ms | 421.8 ms | borderline |
| 6 | 15.6 req/s | 345.2 ms | 690.3 ms | no |
| 8 | 14.4 req/s | 494.8 ms | 982.6 ms | no |
| 10 | 11.1 req/s | 746.9 ms | 1770 ms | no |

**Throughput peaks at ~2 concurrent requests and then falls while latency climbs.** That
shape is the signature of a saturated single-threaded resource: the work is CPU-bound
Python under one GIL, so extra concurrency adds queueing and context-switching without
adding capacity. Past the knee, offering more load makes the service slower *and* less
productive.

**Practical capacity: roughly 30 req/s, provided in-flight concurrency stays at or below
2–3.** Beyond that the latency objective is not met. Note this is a *single process* limit
and the fix is horizontal, not vertical — which runs directly into the Known Gap above.
That is the honest ceiling of this architecture today.

### What has not been measured

- Sustained load over hours (memory growth, MLflow store growth, file-handle leaks).
- Behaviour with `DRIFT_ENABLED=true` under load, where every window forks an Evidently
  interpreter. Expected to reduce throughput; quantifying it is a separate experiment.
- `/prediction_api`, whose latency depends on how many transactions are fraud-flagged
  because only those get a SHAP explanation.
- `/batch_prediction` under any load at all.

---

## Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-09-04 | Two SLIs only (availability, latency) | Every extra SLI is another potential page. CPU/memory are causes, not user-visible symptoms, and belong on a dashboard rather than in an objective. |
| 2026-09-04 | 4xx counts as a *good* event | A 422 from the schema is the service working correctly. Counting it would let a client's malformed payloads burn our budget and page us for their bug. |
| 2026-09-04 | Threshold ratio instead of `histogram_quantile` for latency | The per-handler histogram has only `le`=0.1/0.5/1/+Inf. A computed p99 returns `+Inf` above 1s — blind exactly when it matters. |
| 2026-09-04 | Exclude `/metrics`, `/healthz`, `/readyz` from every SLI | ~10 guaranteed-fast 2xx probe/scrape requests per minute otherwise pin the SLI at 1.0 at demo volumes. |
| 2026-09-04 | Availability target 99.5%, not 99.9% | Single replica, no autoscaling, model warm-up on start. Every rollout is a brief outage, and the budget must absorb the deploys we intend to make. A target the architecture cannot meet is theatre. |
| 2026-09-04 | Record long-window rates directly rather than averaging the 5m ratio | `avg_over_time` of a ratio weights every bucket equally regardless of request count, so one quiet minute with a single error can dominate an hour of healthy traffic. |
| 2026-09-04 | Keep the slow-burn alert despite it being undemonstrable here | The pattern is correct under production load. Documented as never-fired rather than tuned down until it looked impressive. |
| 2026-09-04 | Ship `replicas: 1` and document the gap rather than scaling | Two replicas would make the drift statistic genuinely wrong *and* produce a single alternating Prometheus series. A knowingly-corrupt metric is worse than a documented limitation. |
| 2026-09-04 | TTL-cache the best-run resolution rather than resolving per request | **Measured, not estimated:** 3053ms to resolve vs 17ms to predict — 180x overhead on every request, capping throughput at ~1.4 req/s and putting the latency SLO out of reach. Tradeoff: a new model is adopted within 300s instead of instantly, which is irrelevant at this retrain cadence. |
| 2026-09-04 | Error-rate rules fall back to `0 * <denominator>` when no 5xx exist | Found while verifying: with zero errors the `status="5xx"` selector matches no series, so the ratio was empty and the availability panel went blank *exactly when the service was healthy*. `vector(0)` is label-less and will not join; multiplying the denominator by 0 keeps the `job` label and stays correctly absent under zero traffic. |
| 2026-09-04 | k6 default profile runs *within* measured capacity | A load test that always breaches its thresholds gets ignored. The conformance run gates the SLO; exploring the curve is an explicit manual override. |
