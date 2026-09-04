# Game Day — injected fault exercises

*Deliberately breaking this service on purpose, to find out what the monitoring actually tells you before an outage gets to ask the question.*

**Environment:** local `docker compose` stack
**Run by:** single maintainer
**First run:** 2026-09-04

These are **exercises**, not production incidents. Nothing here describes a real outage affecting real users. The findings are real; the traffic is synthetic.

---

## Progress

- [x] **Game Day 1 — drift worker failure** → wrote [POSTMORTEM-001](POSTMORTEM-001-drift-blindspot.md)
- [ ] Game Day 2 — kill the container mid-request, verify graceful shutdown drains in-flight work
- [ ] Game Day 3 — saturate the AnyIO threadpool, confirm the async probes stay responsive
- [ ] Game Day 4 — fill the disk, observe MLflow store behaviour

---

## Why do this at all

Every alert in `observability/alert_rules.yml` is a *hypothesis*: "if X breaks, this will fire." An untested alert is a hypothesis nobody has checked, and the usual way you find out it was wrong is during the incident it was written for.

A game day converts the hypothesis into an observation. It also tests the parts nobody thinks to test: whether the rule's PromQL matches metrics that actually exist, whether Alertmanager is genuinely reachable, whether the runbook link resolves, and whether the person on the other end can act on the message.

---

## Game Day 1 — the drift worker cannot read its reference

**Hypothesis:** if the drift pipeline fails, the operator finds out.

**Result: FALSE at the time of the exercise.** Every dashboard stayed green while input-drift monitoring was completely dead. See [POSTMORTEM-001](POSTMORTEM-001-drift-blindspot.md).

### Why this injection method

`config.DRIFT_REFERENCE_FILE` is environment-overridable, so pointing it at a non-existent path makes `_require_reference()` raise `FileNotFoundError` **immediately and deterministically**, on every attempt, with no code change and no timing sensitivity.

The alternative — forcing the worker to *hang* — exercises the `subprocess.run(timeout=...)` path instead, but it is slower and less repeatable. It belongs in a separate exercise. Deterministic beats dramatic when the point is to observe the monitoring.

### Procedure

`DRIFT_WINDOW=50` is used throughout so the automatic path triggers inside a short exercise instead of needing 500 requests per window.

```bash
# 0. Baseline: a known-good drift run.
DRIFT_WINDOW=50 docker compose up -d app
python scripts/prime_drift.py     # or replay ~60 rows through /prediction_ui
curl -s localhost:8005/metrics | grep -E '^fraud_(input_)?drift'
# expect: fraud_drift_runs_total{result="success"} 1.0, gauges populated

# 1. Inject.
DRIFT_WINDOW=50 DRIFT_REFERENCE_FILE=/app/prediction_model/monitoring/NOPE.csv \
  docker compose up -d app

# 2. Drive normal traffic and watch what the operator would see.
python scripts/prime_drift.py
curl -s localhost:8005/metrics | grep -E '^fraud_input_drift_(share|dataset_detected)'
curl -s localhost:8005/metrics | grep -E '^fraud_drift_runs_total'

# 3. Follow the alert the whole way.
curl -s localhost:9090/api/v1/alerts        # Prometheus: is it firing?
curl -s localhost:9093/api/v2/alerts        # Alertmanager: did it arrive?
docker compose logs -f webhook-logger       # Receiver: was it delivered?

# 4. Remediate and confirm recovery.
DRIFT_WINDOW=50 docker compose up -d app
python scripts/prime_drift.py
curl -s localhost:8005/metrics | grep -E '^fraud_drift_runs_total'
```

### What was observed

| Signal | During the fault | Reads as |
|---|---|---|
| `POST /prediction_ui` | `200` for every request | healthy |
| `GET /readyz` | `200` | healthy |
| Docker healthcheck | `healthy` | healthy |
| `fraud_input_drift_share` | `0.0` | **"no drift detected"** |
| `fraud_input_drift_dataset_detected` | `0.0` | **"no drift detected"** |
| `fraud_input_drift_window_size` | `0.0` | (unremarkable on a dashboard) |
| `fraud_drift_runs_total{result="failure"}` | `1.0` | **the only true signal** |

Time from injection to a delivered notification: **under 60 seconds** — one scrape interval plus one rule evaluation. `FraudDriftWorkerFailing` reached Alertmanager carrying its `runbook_url`, and the webhook receiver logged the payload.

### Two honest observations from the run

**The alert has a ~15 minute tail.** `increase(fraud_drift_runs_total{result="failure"}[15m]) > 0` keeps firing for up to 15 minutes after the last failure, because the failure is still inside the lookback window. That is correct — the condition really did occur — but a responder who fixes the fault and expects instant green will think the fix did not work. It is written up in the runbook rather than tuned away, because shortening the window would make the alert easier to miss entirely.

**`DRIFT_WINDOW=50` produces a spuriously high drift share** (~0.97 against a 20,000-row reference), which trips `FraudInputDatasetDrift` as a side effect. That is small-sample noise, not real drift, and it is a genuine argument for the production default of 500. Worth knowing before anyone tunes the window down to make demos faster.

### Follow-up actions

Tracked in [POSTMORTEM-001](POSTMORTEM-001-drift-blindspot.md). All landed before this document was written.

---

## Rules for running these

1. **Baseline first.** Capture the healthy state before injecting, or you cannot tell what changed.
2. **One variable.** Inject exactly one fault. Two at once and you learn nothing about either.
3. **Follow the signal the whole way** — metric → rule → Alertmanager → receiver. Most breakage lives at the joins, not in the rules.
4. **Write down what you predicted before you look.** A game day where the monitoring "worked fine" usually means the prediction was formed after the observation.
5. **Always remediate and verify recovery.** An exercise that leaves the system broken is just an outage you scheduled.
6. **Label the write-up as an exercise.** A postmortem that reads like a real outage but was not is a credibility problem waiting to happen.
