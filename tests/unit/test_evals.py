from pathlib import Path

import pytest

from evals.report import previous_report, render
from evals.runner import load_suite

ROOT = Path(__file__).parents[2]


def test_nightly_suite_has_30_unique_tasks_including_smoke() -> None:
    tasks = load_suite(ROOT / "evals" / "tasks.yaml")
    smoke = load_suite(ROOT / "evals" / "smoke.yaml")
    assert len(tasks) == 30 and len({t.id for t in tasks}) == 30
    assert [t.id for t in tasks[:15]] == [t.id for t in smoke]
    login = next(t for t in tasks if t.id == "internet-login-secret")
    assert "{{SECRET:internet_pw}}" in login.task and "internet_pw" in login.secrets


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text(
        "tasks:\n  - {id: x, task: t, allowed_domains: [a.test], expect: {}}\n", encoding="utf-8"
    )
    (tmp_path / "b.yaml").write_text(
        "include: [a.yaml]\ntasks:\n  - {id: x, task: t, allowed_domains: [a.test], expect: {}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="appears twice"):
        load_suite(tmp_path / "b.yaml")


def report(passed: dict[str, bool], cost: float = 0.02) -> dict[str, object]:
    results = [
        {
            "id": k,
            "passed": v,
            "why": "ok" if v else "answer lacks ['x']",
            "status": "done",
            "steps": 3,
            "cost_usd": cost,
            "duration_s": 10.0,
            "result": "r",
        }
        for k, v in passed.items()
    ]
    n = sum(passed.values())
    return {
        "suite": "evals/tasks.yaml",
        "created_at": "20261019T000000Z",
        "models": {"decide": "gpt-4.1", "small": "gpt-4.1-mini"},
        "prompt_version": "system@3",
        "summary": {
            "passed": n,
            "total": len(passed),
            "success_rate": n / len(passed),
            "median_steps": 3,
            "mean_cost_usd": cost,
            "total_cost_usd": cost * len(passed),
            "p95_duration_s": 10.0,
        },
        "results": results,
    }


def test_report_shows_targets_deltas_and_regressions() -> None:
    before = report({"a": True, "b": True, "c": False}, cost=0.03)
    now = report({"a": True, "b": False, "c": True}, cost=0.02)
    md = render(now, before)
    assert "2/3 passed" in md and "MVP target" in md
    assert "**Regressions** (passed last time):" in md and "`b` — answer lacks" in md
    assert "**Fixed since last time:** `c`" in md
    assert "$0.0200 (-0.0100)" in md
    assert render(now).count("Regressions") == 0  # nothing to compare with


def test_previous_report_is_the_latest_earlier_one(tmp_path: Path) -> None:
    for stamp in ("20261017T000000Z", "20261018T000000Z", "20261019T000000Z"):
        (tmp_path / f"tasks-{stamp}.json").write_text("{}", encoding="utf-8")
    (tmp_path / "smoke-20261018T120000Z.json").write_text("{}", encoding="utf-8")
    assert previous_report(tmp_path / "tasks-20261019T000000Z.json") == (
        tmp_path / "tasks-20261018T000000Z.json"
    )
    assert previous_report(tmp_path / "tasks-20261017T000000Z.json") is None


def test_report_warns_when_the_model_changed() -> None:
    before = report({"a": True})
    now = report({"a": True})
    now["provider"] = "anthropic"
    now["models"] = {"decide": "claude-sonnet-4-5", "small": "claude-haiku-4-5"}
    md = render(now, before)
    assert "Provider `anthropic`" in md and "compare different models" in md
    assert "compare different models" not in render(before, report({"a": True}))
