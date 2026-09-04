"""Structured JSON logging for the serving process (Phase 4 SRE).

Stdlib only — no structlog, no python-json-logger. This project has a documented history
of dependency conflicts (evidently/litestar vs python-multipart; dvc vs pathspec), and a
log formatter is not worth another pin.

Why JSON rather than the default format: container stdout is scraped by a log collector
(CloudWatch, Loki, Splunk), and a single-line JSON object is queryable by field —
`level="ERROR" AND logger="prediction_model.monitoring.drift"` — where a formatted string
requires a regex per message shape. It also keeps multi-line tracebacks in ONE record via
the `exc_info` field, instead of the collector splitting a traceback into 30 unrelated
lines, which is how stack traces get lost in aggregation.

Before this existed the only record of a failed drift run was a bare
`print(f"[drift] ...")` with no level, no timestamp and no traceback — invisible to any
level-based filter or alert. See POSTMORTEM-001.
"""
import datetime as _dt
import json
import logging
import os
import sys

# Fields the LogRecord carries that are already represented explicitly below, or that are
# internal bookkeeping. Anything NOT in here is emitted as a custom field, so callers can
# do logger.info("...", extra={"run_id": ...}) and have it land in the JSON.
_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "thread", "threadName", "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            # Explicit UTC with an offset. Container clocks and log-collector clocks
            # disagree often enough that a naive local timestamp makes correlating an
            # incident timeline across services genuinely painful.
            "timestamp": _dt.datetime.fromtimestamp(
                record.created, tz=_dt.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            # Thread name matters here: drift runs execute on the "drift-worker" executor
            # thread, so this alone tells you whether a line came from request handling or
            # from monitoring.
            "thread": record.threadName,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        # default=str so an unexpected non-serialisable value degrades to its repr rather
        # than raising inside the logging call and losing the record entirely.
        return json.dumps(payload, default=str)


def configure_logging(level=None):
    """Install the JSON formatter on the root logger. Idempotent.

    Writes to stdout, not stderr: in a container both are captured, and keeping one stream
    preserves ordering between application logs and uvicorn's access log.
    """
    level = level or os.getenv("LOG_LEVEL", "INFO").upper()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    # Replace rather than append, or uvicorn's default handler emits every record a second
    # time in plain text and the log stream is half JSON, half not.
    root.handlers = [handler]
    root.setLevel(level)

    # uvicorn installs its own handlers at import; hand them back to the root logger so
    # access logs are JSON too.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True

    return root
