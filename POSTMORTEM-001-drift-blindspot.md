# POSTMORTEM-001 — Input-drift monitoring could fail silently

> **This documents a GAME DAY EXERCISE, not a production incident.** The fault was injected deliberately on a local stack against synthetic traffic. No real users were affected, and this service has never run in production. The *defect* it uncovered was real and is now fixed. Procedure: [GAMEDAY.md](GAMEDAY.md).

**Status:** Resolved — all action items landed
**Date of exercise:** 2026-09-04
**Severity:** SEV-2 equivalent (silent loss of a monitoring capability; no user-facing impact)
**Author:** single maintainer
**Related:** [SLO.md](SLO.md) · [RUNBOOK.md](RUNBOOK.md)

---

## Summary

Input-drift monitoring could stop working entirely without producing any signal. Every drift gauge reports *the state of the last successful run*, so when runs failed the gauges simply retained their previous values. On a dashboard, a completely dead drift pipeline was **indistinguishable from a healthy, non-drifting one** — and after a restart it was worse, because the gauges initialise to `0`, which reads as the reassuring "no drift detected".

The game day confirmed the blind spot, and the fix — a run-outcome counter plus an alert on it — closed it. Two adjacent defects found while investigating (an unbounded subprocess call and drift running inside the request's timing path) were fixed at the same time.

## Impact

Exercise only; no user impact.

Had this occurred in production, the impact would be **the loss of a safety net rather than an outage**: predictions keep succeeding, latency is unaffected, and the service looks perfectly healthy. The cost is that genuine input drift — the model being fed transactions that no longer resemble what it was trained on — would go undetected for as long as the drift pipeline stayed broken. The failure is invisible precisely because nothing user-facing breaks.

## Detection

**Nobody detected it, and that is the finding.**

During the exercise the service returned `200` for every prediction, `/readyz` returned `200`, the Docker healthcheck reported `healthy`, and the drift dashboard showed `fraud_input_drift_share = 0.0` — "no drift". The only reason the failure was visible at all is that the exercise author had injected it and knew where to look.

The failure was recorded in exactly one place: a single `print()` on stdout with no log level, no timestamp, and no traceback. Nothing scraped it, nothing alerted on it, and no level-based log filter would have surfaced it.

## Root cause

Three distinct defects, one of which is the headline.

### 1. The metric design could not express failure (primary)

`prediction_model/monitoring/metrics.py` exposed only **gauges**, all of them set by `update_from_summary()` — which is called at the *end* of a successful run. A failing run raises before reaching it, so:

- gauges retain their last successful values → **stale data is indistinguishable from fresh data**;
- on a fresh process the gauges read `0` → **"never measured" is indistinguishable from "measured, no drift"**.

`safe_run_drift_check()` then swallowed the exception by design — correct, because monitoring must never break serving, but it meant the failure had no path to reach anyone.

There was no counter, no run-outcome metric, and no staleness signal. **The system had no way to represent "the monitoring itself is broken."**

### 2. The drift subprocess had no timeout

`drift.compute_drift()` called `subprocess.run(...)` with no `timeout=`. A wedged Evidently worker would block forever, consuming its thread permanently, with no metric and no log line while it hung.

This matters more than it first appears, because of how the work was scheduled:

> `subprocess.run` with no timeout → Starlette runs a sync `BackgroundTask` via `run_in_threadpool`, i.e. the **shared AnyIO threadpool** (default 40 slots) → FastAPI dispatches **every `def` endpoint into that same pool**, which in `main.py` is every endpoint except `batch_predict` → N hung workers permanently consume N of 40 slots → at 40, every sync endpoint queues → **a sync health endpoint would queue too, the liveness probe would time out, and the kubelet would restart a pod that was merely saturated.**

One missing keyword argument, and the blast radius is a restart loop.

### 3. Drift ran inside the request's measured latency

Starlette awaits `BackgroundTask` *inside* the response object (`starlette/responses.py`: `await self.background()` after the body is sent), which is still inside the Prometheus middleware's timing block (`middleware.py`, duration recorded in a `finally:`).

So every 500th request recorded its own serving time **plus the full Evidently subprocess wall time** — latency no client ever experienced. Against a 99%-under-500ms objective, that one-in-500 phantom would burn a meaningful share of the latency budget measuring nothing real.

**Contributing factor:** the `for: 10m` on the existing drift alerts is fine for drift itself, but there was no alert of any kind on the health of the pipeline producing those numbers. The monitoring was monitored by nothing.

## Timeline

All times from the exercise on 2026-09-04.

| Time | Event |
|---|---|
| T+0 | Baseline captured: one successful drift run, `fraud_drift_runs_total{result="success"} = 1`, gauges populated (`share = 0.966`, `window_size = 50`) |
| T+0 | **Fault injected:** app restarted with `DRIFT_REFERENCE_FILE` pointing at a non-existent path |
| T+1 | 60 predictions replayed through `/prediction_ui` — **all returned `200`** |
| T+2 | Drift gauges read `share = 0.0`, `dataset_detected = 0.0`. Dashboard green. `/readyz` `200`. Container `healthy` |
| T+2 | `fraud_drift_runs_total{result="failure"}` increments to `1` — the only true signal |
| T+3 | `FraudDriftWorkerFailing` fires in Prometheus (`for: 0m`, one evaluation cycle) |
| T+3 | Alert reaches Alertmanager, grouped, carrying its `runbook_url` |
| T+3 | Notification delivered to the webhook receiver — **under 60s from injection** |
| T+4 | Remediated: correct reference path restored, traffic replayed |
| T+5 | `fraud_drift_runs_total{result="success"}` increments, gauges repopulate |
| T+5 | Alert continues firing for the remainder of its 15m lookback window, then clears |

## What went well

- The counter-based alert fired within **one evaluation cycle**, with no staleness heuristic and no tuning.
- The whole path — metric → recording/alert rule → Alertmanager → receiver — worked on the first attempt, including alert grouping and the runbook link.
- The failure degraded exactly as intended: **monitoring broke, serving did not.** Every prediction succeeded throughout. The original decision to swallow drift exceptions was right; it was only ever missing a signal.
- Restoring the reference path recovered the pipeline with no manual intervention beyond the restart.

## What went badly

- The blind spot existed at all. Drift monitoring was built in Phase 3b and shipped with **no way to tell whether it was working** — a monitoring system with no monitoring of its own.
- A "green" drift dashboard was actively misleading. Worse than no dashboard, because it invites confidence.
- The one record of failure was a bare `print()`: no level, no timestamp, no traceback, invisible to any log filter.
- The alert rules had been written but never *exercised*. Every one was an untested hypothesis until this game day.

## Where we got lucky

The exercise used a missing reference file, which fails **immediately**. Had the first real failure been a *hang* instead, the missing `subprocess.run` timeout would have made it far worse: threads consumed permanently, no error at all (not even the `print`), and eventually threadpool starvation taking the sync endpoints down with it. The benign failure mode surfaced the design flaw before the malignant one could.

## Action items

| # | Action | Type | Status |
|---|---|---|---|
| 1 | Add `fraud_drift_runs_total{result}` counter, pre-initialised so both label values exist at `0` from the first scrape | detect | **Done** — `metrics.py` |
| 2 | Alert `FraudDriftWorkerFailing` on `increase(...{result="failure"}[15m]) > 0` | detect | **Done** — `alert_rules.yml` |
| 3 | Add `FraudDriftStale` as a backstop, with a `> 0` guard so it does not fire against the epoch on every fresh start | detect | **Done** — `alert_rules.yml` |
| 4 | Bound the worker: `subprocess.run(..., timeout=config.DRIFT_WORKER_TIMEOUT)` | mitigate | **Done** — `drift.py` |
| 5 | Move drift off `BackgroundTask` onto a dedicated `ThreadPoolExecutor(max_workers=1)` — off the timed request path, and capped at one thread | mitigate | **Done** — `drift.py`, `main.py` |
| 6 | Count outcomes inside `run_drift_check()` so the manual `POST /monitoring/drift` path is counted too, not just the scheduled one | detect | **Done** — `drift.py` |
| 7 | Replace the bare `print` with structured JSON logging carrying level, timestamp and traceback | detect | **Done** — `logging_config.py` |
| 8 | Wire Alertmanager so rules reach a receiver at all | respond | **Done** — `docker-compose.yml`, `alertmanager.yml` |
| 9 | Add `runbook_url` to every rule, resolving to a real runbook entry | respond | **Done** — `alert_rules.yml`, `RUNBOOK.md` |
| 10 | Regression test that drift is not scheduled as a `BackgroundTask`, and that the probes stay `async def` | prevent | **Done** — `tests/test_drift.py`, `tests/test_health.py` |
| 11 | Inhibit `FraudDriftStale` while `FraudDriftWorkerFailing` is active — one fault should not produce two notifications | respond | **Done** — `alertmanager.yml` |
| 12 | Game day for the *hang* path, exercising the new timeout rather than the missing-file path | prevent | Open — [GAMEDAY.md](GAMEDAY.md) Game Day 3 |

## Lessons

**A gauge cannot tell you it is stale.** Any metric that means "state at the last successful update" needs a companion counter or timestamp, or its most dangerous failure mode — quietly freezing — is invisible by construction. This generalises well beyond drift: it applies to every cached, batch-computed or externally-populated gauge in any system.

**Swallowing an exception is a decision to lose information, so pay for it deliberately.** Catching everything in `safe_run_drift_check()` was correct — monitoring must not break serving. The mistake was catching it and then not emitting a metric. The catch is fine; the silence was the defect.

**Untested alerts are hypotheses.** Every rule here was written months before it was ever exercised, and the exercise is what proved that the *most important failure mode had no rule at all*. Writing the alert is not the same as knowing it fires.

**Watch what runs inside your instrumentation boundary.** The BackgroundTask latency defect was invisible in code review — the call looked asynchronous and the docstring said "off the request latency path". It was only visible by reading Starlette's source to find where `await self.background()` actually sits relative to the middleware's timer. Framework defaults determine what your metrics mean.
