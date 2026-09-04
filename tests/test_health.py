"""Probe-endpoint tests (Phase 4 SRE).

Two deliberate choices here:

1. This is the ONLY test module that imports `main`. Importing it executes
   `Instrumentator().instrument(app).expose(app)` at module scope, which registers
   collectors on the global Prometheus REGISTRY that test_drift.py asserts against.

2. The handlers are called directly rather than through `fastapi.testclient.TestClient`.
   TestClient is unusable in this environment: starlette 0.27.0 constructs
   `httpx.Client(app=...)` (testclient.py:399), and httpx 0.28 removed that shortcut in
   favour of an explicit ASGITransport. Pinning httpx down is not worth it in a project
   with this dependency history, and calling the coroutines directly still exercises the
   real readiness logic and the real response objects.
"""

import asyncio
import inspect

import pytest

import main


def _call(coro_fn):
    """Run an async path operation to completion and return its result."""
    return asyncio.run(coro_fn())


def test_healthz_is_ok_regardless_of_model_state(monkeypatch):
    """Liveness must not depend on the model. A pod with no model should stay alive
    (and be pulled from the Service by readiness) rather than crash-loop: restarting
    cannot conjure an absent artifact."""
    monkeypatch.setattr(main, "is_model_ready", lambda: False)
    assert _call(main.healthz)["status"] == "ok"


def test_readyz_503_when_model_absent(monkeypatch):
    monkeypatch.setattr(main, "is_model_ready", lambda: False)
    resp = _call(main.readyz)
    assert resp.status_code == 503


def test_readyz_200_when_model_loaded(monkeypatch):
    monkeypatch.setattr(main, "is_model_ready", lambda: True)
    result = _call(main.readyz)
    # The ready path returns a plain dict (FastAPI serialises it as 200), not a Response.
    assert result == {"status": "ready", "model_loaded": True}


def test_probes_are_async():
    """Load-bearing design property, not style.

    A sync (`def`) probe is dispatched into the shared AnyIO threadpool — the same pool
    every other `def` endpoint and every sync BackgroundTask uses. Under saturation the
    probe queues, liveness times out, and the kubelet restarts a pod that was merely
    busy, converting a latency problem into an availability one. Keep these coroutines.
    """
    routes = {r.path: r.endpoint for r in main.app.routes if hasattr(r, "endpoint")}
    for path in ("/healthz", "/readyz"):
        assert inspect.iscoroutinefunction(routes[path]), f"{path} must be async def"


def test_probes_are_registered():
    paths = {r.path for r in main.app.routes if hasattr(r, "endpoint")}
    assert {"/healthz", "/readyz"} <= paths


def test_probes_do_not_touch_the_drift_buffer(monkeypatch):
    """Probe traffic must not pollute the drift window, or the input-drift statistic
    becomes a mixture of real transactions and health checks."""
    monkeypatch.setattr(main, "is_model_ready", lambda: True)
    from prediction_model.monitoring import drift

    before = len(drift._BUFFER)
    _call(main.healthz)
    _call(main.readyz)
    assert len(drift._BUFFER) == before


def test_is_model_ready_does_no_mlflow_io(monkeypatch):
    """A readiness probe runs ~6/min/pod forever. If it resolved the best run it would
    scan the whole MLflow file store on every call."""
    from prediction_model import predict

    def _boom(*a, **k):  # pragma: no cover - must never be reached
        raise AssertionError("is_model_ready() must not resolve runs or load models")

    monkeypatch.setattr(predict, "_resolve_best_run_id", _boom)
    monkeypatch.setattr(predict, "_get_model_and_explainer", _boom)
    assert predict.is_model_ready() in (True, False)


def test_warm_model_is_non_fatal(monkeypatch):
    """Warmup failure must degrade to not-ready, never propagate. A serving process that
    cannot reach MLflow should come up live-but-unready, not crash-loop."""
    from prediction_model import predict

    def _boom(*a, **k):
        raise AttributeError("simulating mlflow.get_experiment_by_name() -> None")

    monkeypatch.setattr(predict, "_get_model_and_explainer", _boom)
    assert predict.warm_model() is False


def test_starlette_can_reach_anyio_to_thread():
    """Regression guard for a transitive-dependency break that took the service down.

    starlette 0.27.0's run_in_threadpool does `import anyio` and then accesses
    `anyio.to_thread` as an ATTRIBUTE. anyio 4.15.0 removed that lazy attribute, so every
    sync (`def`) endpoint raised AttributeError and returned 500 — while /healthz and
    /readyz, being async, kept returning 200 and the container kept reporting healthy.

    Nothing in this repo changed to cause it; anyio was simply unpinned and published a
    new release. This asserts the exact access pattern starlette uses, so a future
    unpinning fails here instead of in production.
    """
    import anyio
    assert hasattr(anyio, "to_thread"), (
        "starlette's run_in_threadpool needs anyio.to_thread as an attribute; "
        "pin anyio (see requirements.txt)"
    )
    assert callable(anyio.to_thread.run_sync)


def test_run_resolution_is_ttl_cached(monkeypatch):
    """Measured: resolving the best run cost 3053ms vs 17ms to actually predict — 180x,
    paid on every request. Without this cache the 500ms latency SLO is unreachable."""
    from prediction_model import predict

    calls = []
    monkeypatch.setattr(predict, "_search_best_run_id", lambda: calls.append(1) or "run-abc")
    monkeypatch.setattr(predict, "_RESOLVE_CACHE", {"run_id": None, "at": 0.0})

    assert predict._resolve_best_run_id() == "run-abc"
    for _ in range(10):
        predict._resolve_best_run_id()
    assert len(calls) == 1, "run resolution must not hit MLflow on every request"


def test_run_resolution_refreshes_after_ttl(monkeypatch):
    """The cache must expire, or a newly trained better model is never picked up."""
    import time as _time
    from prediction_model import predict

    calls = []
    monkeypatch.setattr(predict, "_search_best_run_id", lambda: calls.append(1) or "run-abc")
    monkeypatch.setattr(predict, "_RESOLVE_CACHE", {"run_id": None, "at": 0.0})
    monkeypatch.setattr(predict, "_RESOLVE_TTL", 0.05)

    predict._resolve_best_run_id()
    _time.sleep(0.08)
    predict._resolve_best_run_id()
    assert len(calls) == 2, "cache must re-resolve once the TTL expires"
