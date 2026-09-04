# Multi-stage: base (runtime deps) -> test (adds pytest, runs the suite) -> runtime.
#
# The point of the split is that pytest and the test suite never reach the shipped image,
# while a failing test still fails the build. See the COPY --from=test note below for the
# trap that makes that actually work.

FROM python:3.10-slim-bookworm AS base

WORKDIR /app

# PYTHONUNBUFFERED is not cosmetic in a container: stdout to a pipe is block-buffered by
# default, so logs arrive in 4KB chunks and the last partial buffer is LOST on a crash —
# exactly the output you need when diagnosing why the process died.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- test stage ------------------------------------------------------------------------
FROM base AS test

COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt

COPY . /app

# The stamp file exists so the runtime stage can depend on this stage. Without that
# dependency BuildKit prunes `test` as unreachable and the suite silently never runs —
# a green build that tested nothing.
RUN pytest -v tests/ && echo "passed" > /tests-passed


# --- runtime stage ---------------------------------------------------------------------
FROM base AS runtime

# Forces the test stage into the build graph. Cheap file, load-bearing line.
COPY --from=test /tests-passed /tests-passed

COPY . /app

# uid/gid 1000 matches runAsUser/runAsGroup in deployment.yml. If these drift apart the
# pod starts but every write fails with EACCES.
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --no-create-home --shell /usr/sbin/nologin appuser \
    # config.BATCH_OUTPUT_DIR resolves to /app/batch_predictions, and .dockerignore keeps
    # it out of the image, so it does not exist here. Creating it as root and handing it
    # over is required: a non-root process cannot mkdir inside root-owned 0755 /app, and
    # POST /batch_prediction would 500 on PermissionError — a regression the root image
    # did not have.
    && mkdir -p /app/batch_predictions \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8005

# python:*-slim ships NEITHER curl NOR wget, so the usual `curl -f` healthcheck is
# permanently "unhealthy". urllib is already in the interpreter we are shipping.
#
# Targets /healthz (liveness), never /readyz: a readiness-based healthcheck combined with
# `depends_on: {condition: service_healthy}` would block the whole compose stack whenever
# the model is unavailable, which is precisely the failure we want to be able to observe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8005/healthz', timeout=3).status == 200 else 1)"]

ENTRYPOINT ["python"]

CMD ["main.py"]
