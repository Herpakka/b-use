"""Prometheus metrics, scraped per replica at ``/metrics``.

The plan's four: success rate (``bu_runs_finished_total`` by status), p50/p95 duration
(``bu_run_duration_seconds``), cost per run (``bu_run_cost_usd``), pool utilisation
(``bu_browser_sessions`` / ``bu_browser_capacity``). Example PromQL is in docs/runbook.md.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from core.store import RunRecord

RUNS_STARTED = Counter("bu_runs_started_total", "Runs admitted")
RUNS_REFUSED = Counter("bu_runs_refused_total", "Run requests refused", ["reason"])
RUNS_FINISHED = Counter("bu_runs_finished_total", "Runs that ended", ["status"])
RUNS_ADOPTED = Counter("bu_runs_adopted_total", "Runs taken over from another replica")
RUN_DURATION = Histogram(
    "bu_run_duration_seconds",
    "Wall-clock time from creation to the end (including waiting for a human)",
    buckets=(5, 10, 20, 30, 45, 60, 90, 120, 180, 300, 600, 1800),
)
RUN_COST = Histogram(
    "bu_run_cost_usd",
    "LLM cost of a finished run",
    buckets=(0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.6, 1.0),
)
RUN_STEPS = Histogram(
    "bu_run_steps", "Steps of a finished run", buckets=(1, 2, 3, 5, 8, 12, 16, 25, 50, 100)
)
COST_TOTAL = Counter("bu_llm_cost_usd_total", "LLM spend of finished runs")
TOKENS_TOTAL = Counter("bu_llm_tokens_total", "LLM tokens of finished runs", ["direction"])
HUMAN_REQUESTS = Counter("bu_human_requests_total", "Pauses for a human", ["kind"])
HUMAN_ANSWERS = Counter("bu_human_answers_total", "Human answers", ["approved"])

RUNS_ACTIVE = Gauge("bu_runs_active", "Runs this replica owns", ["state"])
BROWSER_UP = Gauge("bu_browser_up", "1 if the browser answers its health probe")
BROWSER_SESSIONS = Gauge("bu_browser_sessions", "Open browser contexts")
BROWSER_CAPACITY = Gauge("bu_browser_capacity", "Browser contexts this replica can hold")
BROWSER_RESTARTS = Gauge("bu_browser_restarts", "Browser restarts since the replica started")


def run_finished(run: RunRecord) -> None:
    RUNS_FINISHED.labels(run.status).inc()
    if run.finished_at is not None:
        RUN_DURATION.observe(max(0.0, (run.finished_at - run.created_at).total_seconds()))
    RUN_COST.observe(run.cost_usd)
    RUN_STEPS.observe(run.steps)
    COST_TOTAL.inc(run.cost_usd)
    TOKENS_TOTAL.labels("in").inc(run.tokens_in)
    TOKENS_TOTAL.labels("out").inc(run.tokens_out)
