# POSTMORTEM-002 — An unpinned transitive dependency broke every sync endpoint

> **Not a game day.** Unlike [POSTMORTEM-001](POSTMORTEM-001-drift-blindspot.md), nobody injected this. It was found by rebuilding the container image during unrelated work, and it would have shipped a broken image to the registry. No production deployment exists, so no users were affected — but the defect and the discovery were both real.

**Status:** Resolved
**Date:** 2026-09-04
**Severity:** SEV-1 equivalent (total failure of all prediction endpoints)
**Author:** single maintainer

---

## Summary

`anyio` — a dependency this project never names, reached via `fastapi → starlette → anyio` — published version 4.15.0, which removed the lazy `anyio.to_thread` attribute that `starlette==0.27.0` accesses in `run_in_threadpool`. Because `anyio` was unpinned, the next image build picked it up.

The result: **every synchronous (`def`) endpoint returned HTTP 500**, including all three prediction endpoints. No code in this repository changed. The last green build and the first broken build were built from identical source.

## Impact

Complete loss of prediction capability in any freshly built image:

- `POST /prediction_api`, `POST /prediction_ui`, `GET /`, `GET /monitoring/drift` — all `500`
- `AttributeError: module 'anyio' has no attribute 'to_thread'` on every request

No user impact: the service is not deployed. But CI builds and pushes to GHCR on every push to `main`, so **the broken image would have been published and tagged `:latest`.**

## Detection

Found manually, by running `docker compose up` after a rebuild and reading the logs. That is luck, not process.

### Why the health checks did not catch it

This is the interesting part, and it is a direct consequence of a design decision made earlier the same day.

`/healthz` and `/readyz` were deliberately written as `async def` so they run on the event loop and never queue behind saturated threadpool work (see the threadpool starvation chain in [POSTMORTEM-001](POSTMORTEM-001-drift-blindspot.md)). That decision is correct and stays.

But it means the probes ran on **the exact code path that still worked**. While every `def` endpoint was failing:

| Check | Result |
|---|---|
| `GET /healthz` | `200` — process alive. **True.** |
| `GET /readyz` | `200` — model loaded. **True.** |
| Docker `HEALTHCHECK` | `healthy` |
| Kubernetes liveness/readiness | would both pass |

**Every health check was correct and every one was useless here.** Liveness answers "is the process alive"; readiness answers "is the model loaded". Neither claims "the service can serve requests", and it is a mistake to read them as if they did.

The signal that *would* have caught it is the availability SLI: an instant 100% 5xx rate on every real handler, tripping `FraudAPIErrorBudgetFastBurn` within minutes. That is the argument for having SLIs at all — probes check components, SLIs check the user's experience.

## Root cause

`requirements.txt` pinned 17 packages by exact version but did not pin their transitive dependencies. `anyio` was therefore resolved fresh at every image build.

`starlette==0.27.0` does:

```python
import anyio
...
return await anyio.to_thread.run_sync(func, *args)   # attribute access, no explicit import
```

`anyio` 4.15.0 dropped the lazy `__getattr__` shim that made `anyio.to_thread` resolvable without `import anyio.to_thread`. The submodule still exists — `import anyio.to_thread` works fine — but the attribute access starlette relies on raises.

Contributing factors:

1. **The previously built image masked it for weeks.** It had an older `anyio` baked into its layers, so local `docker compose` runs stayed green while a fresh build would have failed. The bug was latent in `requirements.txt`, not in any image anyone was running.
2. **The test suite could not catch it.** `fastapi.testclient.TestClient` is unusable in this environment for an unrelated reason (starlette 0.27 passes `app=` to `httpx.Client`, which httpx 0.28 removed), so no test exercises a real request through the ASGI stack.
3. **The host venv had `anyio` 4.14.1** and kept passing, so `pytest` locally gave a false all-clear.

## Timeline

| Time | Event |
|---|---|
| — | `anyio` 4.15.0 published upstream |
| T+0 | Image rebuilt during unrelated hardening work |
| T+1 | `docker compose up` — container reports `healthy`, both probes `200` |
| T+2 | `GET /` returns `500`; logs show `AttributeError: module 'anyio' has no attribute 'to_thread'` |
| T+3 | Versions compared: container `anyio` 4.15.0, host venv 4.14.1 — same source, different resolution |
| T+4 | `anyio==4.14.1` pinned in `requirements.txt` with the reasoning inline |
| T+5 | Rebuilt; `POST /prediction_ui` returns `200` with a verdict |
| T+6 | Regression test added asserting starlette's exact access pattern |

## Action items

| # | Action | Type | Status |
|---|---|---|---|
| 1 | Pin `anyio==4.14.1` explicitly, with a comment explaining that it is transitive and why it is pinned | mitigate | **Done** |
| 2 | Regression test asserting `hasattr(anyio, "to_thread")` and that `run_sync` is callable — starlette's exact access pattern | prevent | **Done** — `tests/test_health.py` |
| 3 | Pin `prometheus_fastapi_instrumentator==6.1.0` (also unpinned; the SLO queries depend on its exact metric names and label sets) | prevent | **Done** |
| 4 | Document that probes verify components, not end-to-end serving, and that the availability SLI is the real detector | prevent | **Done** — `RUNBOOK.md`, this document |
| 5 | Add a smoke test that exercises one real request through the ASGI stack | prevent | Open — blocked on the starlette/httpx `TestClient` incompatibility; needs an explicit `ASGITransport` or a dependency upgrade |
| 6 | Fully pin the dependency tree (`pip-compile` / hash-pinned lockfile) so transitive resolution is reproducible | prevent | Open — the correct fix; item 1 only closes this one instance |
| 7 | Add a CI smoke step that starts the built image and asserts a real prediction returns 200 | prevent | Open — CI currently tests source, then builds and pushes without exercising the artifact |

## Lessons

**Pinning your direct dependencies is not pinning your dependencies.** Seventeen exact pins gave a convincing illusion of reproducibility while the actual failure arrived through a package the project never mentions. Either lock the full tree or accept that every build is a fresh roll of the dice.

**A test suite that passes in a different environment than it ships in is a partial test suite.** The host venv and the image resolved different versions from the same file, and the host was the one being tested. The suite was green and the artifact was broken.

**Health checks are not smoke tests, and conflating them is dangerous.** Both probes reported the literal truth — the process was alive, the model was loaded — while the service could not serve a single prediction. `/healthz` and `/readyz` deliberately avoid the request-handling path so they cannot be starved by it; the cost of that correctness is that they cannot detect a broken request-handling path. That trade is right, but only if something else covers the gap. Here, that something is the availability SLI.

**Item 7 is the one that would actually have caught it.** CI tests the source and then builds and pushes an artifact it never runs. Testing what you ship, rather than what you built it from, is the difference between the two.
