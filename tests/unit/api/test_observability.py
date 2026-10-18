"""5.3/5.4: Prometheus metrics per replica, and the Langfuse tracer's per-run config."""

from pathlib import Path

from pydantic import SecretStr

from agent.tracing import TRACE_NAME, Tracer, make_tracer
from tests.agent_fakes import PLAN, FakeBrowser, tool
from tests.api_support import SHOP, client_for, memory_factory, start_run, status_in, wait_for
from tests.conftest import make_settings


def metric(text: str, name: str, default: float | None = None) -> float:
    """Value of the sample called exactly ``name`` (labels included)."""
    for line in text.splitlines():
        if line.split(" ")[0] == name:
            return float(line.split()[-1])
    if default is not None:
        return default
    raise AssertionError(f"{name} not in metrics")


def test_metrics_cover_success_duration_cost_and_pool() -> None:
    factory = memory_factory(FakeBrowser(SHOP), [PLAN, tool("Done", result="ok")])
    with client_for(factory) as client:
        before = client.get("/metrics").text
        done_before = metric(before, 'bu_runs_finished_total{status="done"}', 0.0)
        run = start_run(client)
        wait_for(client, run["id"], status_in("done"))
        text = client.get("/metrics").text
    assert metric(text, 'bu_runs_finished_total{status="done"}') == done_before + 1
    assert "bu_run_duration_seconds_bucket" in text and "bu_run_cost_usd_sum" in text
    assert metric(text, "bu_browser_capacity") == 4
    assert metric(text, "bu_browser_up") == 1
    assert 'bu_runs_active{state="running"}' in text
    assert "/metrics" not in client.get("/openapi.json").text


def test_tracing_is_off_without_keys_and_tags_each_run() -> None:
    assert make_tracer(make_settings(Path("unused"))) is None
    settings = make_settings(
        Path("unused"),
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key=SecretStr("sk-lf-test"),
        langfuse_host="http://127.0.0.1:9",  # nothing listens: spans are just dropped
    )
    tracer = Tracer(settings)
    try:
        config = tracer.config("run-123", user="acme", tags=["api"])
        assert config["run_name"] == TRACE_NAME
        assert config["metadata"] == {
            "langfuse_session_id": "run-123",
            "langfuse_tags": ["api"],
            "langfuse_user_id": "acme",
        }
        assert len(config["callbacks"]) == 1
        assert tracer._mask(data={"q": "mail bob@example.com"}) == {"q": "mail [email]"}
    finally:
        tracer.client.shutdown()
