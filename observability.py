"""Observability wiring for the PR multi-agent system.

Everything here is self-hostable — no cloud-only services:

  - Logging:  structlog -> JSON on stdout. Ship it anywhere you like
              (Loki, Vector, Fluent Bit, a plain file) with zero code changes.
  - Metrics:  prometheus_client -> exposes /metrics for a self-hosted
              Prometheus to scrape; graph it in a self-hosted Grafana.
  - Tracing:  Langfuse, run via their official self-host docker-compose
              (https://github.com/langfuse/langfuse -> /docker-compose.yml).
              Wired in via LangChain's standard callback handler, so every
              LLM call's prompt/response/latency/token-usage is captured
              per agent, per PR. If LANGFUSE_* env vars aren't set, tracing
              is simply skipped — nothing else is affected.

Nothing in this module talks to any external network service by default.
"""
import logging
import os
import time
from contextlib import contextmanager
from typing import Optional

import structlog
from prometheus_client import Counter, Histogram, start_http_server

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------


def configure_logging(level: str = "INFO") -> None:
    """Call once at process startup. Renders JSON lines to stdout."""
    logging.basicConfig(format="%(message)s", level=level, force=True)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

AGENT_RUNS = Counter(
    "pr_agent_runs_total", "Agent node executions", ["agent", "status"]
)
AGENT_LATENCY = Histogram(
    "pr_agent_latency_seconds", "Latency of each agent node", ["agent"]
)
PIPELINE_RUNS = Counter(
    "pr_pipeline_runs_total", "Full pipeline executions", ["status"]
)
PIPELINE_LATENCY = Histogram(
    "pr_pipeline_latency_seconds", "End-to-end pipeline latency"
)
RISK_LEVEL_TOTAL = Counter(
    "pr_risk_level_total", "Risk levels assigned by risk_agent", ["risk_level"]
)
EMAIL_SEND_TOTAL = Counter(
    "pr_email_send_total", "Email send attempts by email_agent", ["status"]
)
LLM_CALLS_TOTAL = Counter(
    "pr_llm_calls_total", "LLM invocations", ["agent"]
)


def start_metrics_server(port: int = 8000) -> None:
    """Expose /metrics on `port` for a self-hosted Prometheus to scrape."""
    start_http_server(port)
    log.info("metrics_server_started", port=port)


# ---------------------------------------------------------------------------
# Agent instrumentation
# ---------------------------------------------------------------------------


@contextmanager
def observe_agent(agent_name: str, **context):
    """Wrap an agent node body: structured logs + Prometheus metrics + timing.

    Usage:
        with observe_agent("fetch_agent", repo=repo, pr_number=pr_number):
            ... agent body ...
    """
    start = time.perf_counter()
    log.info("agent_started", agent=agent_name, **context)
    status = "success"
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - re-raised after logging
        status = "error"
        log.error("agent_failed", agent=agent_name, error=str(exc), **context)
        raise
    finally:
        duration = time.perf_counter() - start
        AGENT_LATENCY.labels(agent=agent_name).observe(duration)
        AGENT_RUNS.labels(agent=agent_name, status=status).inc()
        log.info(
            "agent_finished",
            agent=agent_name,
            status=status,
            duration_seconds=round(duration, 3),
            **context,
        )


@contextmanager
def observe_pipeline(**context):
    """Wrap the full app.invoke(...) call: structured logs + Prometheus metrics."""
    start = time.perf_counter()
    log.info("pipeline_started", **context)
    status = "success"
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        status = "error"
        log.error("pipeline_failed", error=str(exc), **context)
        raise
    finally:
        duration = time.perf_counter() - start
        PIPELINE_LATENCY.observe(duration)
        PIPELINE_RUNS.labels(status=status).inc()
        log.info(
            "pipeline_finished",
            status=status,
            duration_seconds=round(duration, 3),
            **context,
        )


def record_llm_call(agent_name: str) -> None:
    LLM_CALLS_TOTAL.labels(agent=agent_name).inc()


def record_risk_level(risk_level: str) -> None:
    RISK_LEVEL_TOTAL.labels(risk_level=risk_level).inc()


def record_email_result(success: bool) -> None:
    EMAIL_SEND_TOTAL.labels(status="success" if success else "error").inc()


# ---------------------------------------------------------------------------
# Langfuse tracing (self-hosted)
# ---------------------------------------------------------------------------


def get_langfuse_handler(trace_name: Optional[str] = None, **trace_metadata):
    """Return a Langfuse LangChain callback handler, or None if unconfigured.

    Point LANGFUSE_HOST at your self-hosted instance, e.g. http://localhost:3000
    (see https://github.com/langfuse/langfuse for the self-host docker-compose).
    Returns None — a safe no-op — if the env vars aren't set or the package
    isn't installed, so tracing is opt-in and never blocks the pipeline.
    """
    host = os.getenv("LANGFUSE_HOST")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not all([host, public_key, secret_key]):
        return None

    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        log.warning("langfuse_not_installed", hint="pip install langfuse")
        return None

    return CallbackHandler(
        host=host,
        public_key=public_key,
        secret_key=secret_key,
        trace_name=trace_name,
        metadata=trace_metadata or None,
    )