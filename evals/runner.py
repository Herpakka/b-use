"""Run a YAML task suite against the real agent and report success / steps / cost / duration.

uv run python -m evals.runner evals/tasks.yaml [--only books-price] [--min-rate 0.7]
    [--har record|replay|replay-strict] [--summary-md $GITHUB_STEP_SUMMARY]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import secrets
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from agent.cli import print_event
from agent.context import AgentContext
from agent.graph import compile_graph
from agent.llm import make_chat_model
from agent.prompts import prompt_version
from agent.runner import HumanHandler, drive, initial_state, limits_from, open_checkpointer
from agent.tracing import make_tracer
from browser.local import LocalBrowserService
from browser.pool import BrowserPool
from core.config import Settings, StaticSecret, get_settings
from core.costs import PriceTable
from core.logging import configure_logging
from core.secrets import Secret, StaticSecretStore
from evals import report as report_md
from evals.har import HarBrowserService, HarMode


class Expect(BaseModel):
    contains: list[str] = Field(default_factory=list)
    any_of: list[str] = Field(default_factory=list)
    regex: str | None = None
    url_contains: str | None = None


class EvalTask(BaseModel):
    id: str
    task: str
    allowed_domains: list[str]
    start_url: str | None = None
    expect: Expect
    human: str = "deny"
    max_steps: int | None = None
    secrets: dict[str, StaticSecret] = Field(default_factory=dict)


class Suite(BaseModel):
    include: list[str] = Field(default_factory=list)  # other suites, relative to this file
    tasks: list[EvalTask] = Field(default_factory=list)


def load_suite(path: Path) -> list[EvalTask]:
    suite = Suite.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    tasks = [t for inc in suite.include for t in load_suite(path.parent / inc)] + suite.tasks
    seen: set[str] = set()
    for t in tasks:
        if t.id in seen:
            raise ValueError(f"{path}: task id {t.id!r} appears twice")
        seen.add(t.id)
    return tasks


@dataclass
class TaskResult:
    id: str
    passed: bool
    why: str
    status: str
    steps: int
    cost_usd: float
    duration_s: float
    result: str | None


def check(expect: Expect, state: dict[str, Any]) -> tuple[bool, str]:
    if state.get("status") != "done":
        return False, f"status {state.get('status')}: {state.get('failure_reason')}"
    text = str(state.get("result") or "")
    low = text.lower()
    missing = [c for c in expect.contains if c.lower() not in low]
    if missing:
        return False, f"answer lacks {missing}"
    if expect.any_of and not any(a.lower() in low for a in expect.any_of):
        return False, f"answer has none of {expect.any_of}"
    if expect.regex and not re.search(expect.regex, text):
        return False, f"answer does not match /{expect.regex}/"
    if expect.url_contains and expect.url_contains not in str(state.get("current_url") or ""):
        return False, f"ended on {state.get('current_url')}"
    return True, "ok"


def responder(policy: str) -> HumanHandler:
    async def respond(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("kind") == "question":
            answer = (
                policy
                if policy not in ("approve", "deny")
                else "No human is available during this evaluation; continue on your own."
            )
            return {"approved": True, "answer": answer}
        if policy == "approve":
            return {"approved": True, "answer": ""}
        return {"approved": False, "answer": "No human is available during this evaluation."}

    return respond


async def run_suite(
    tasks: list[EvalTask],
    settings: Settings,
    *,
    memory: bool,
    verbose: bool,
    har: HarMode | None = None,
    har_dir: Path = Path("evals/har"),
    suite_name: str = "suite",
) -> list[TaskResult]:
    prices = PriceTable.from_settings(settings)
    prices.require(settings.model_decide, settings.model_small)
    results: list[TaskResult] = []
    tag = secrets.token_hex(3)
    static = StaticSecretStore(settings.local_secrets)
    task_secrets: dict[str, Secret] = {}

    async def resolve(scope: str, name: str) -> Secret | None:
        return task_secrets.get(name) or await static.get(scope, name)

    async with (
        BrowserPool(settings) as pool,
        open_checkpointer(settings, memory=memory) as saver,
    ):
        browser = HarBrowserService(pool, har, har_dir) if har else None
        ctx = AgentContext(
            browser=browser or LocalBrowserService(pool),
            llm_decide=make_chat_model(settings, settings.model_decide),
            llm_small=make_chat_model(settings, settings.model_small),
            settings=settings,
            prices=prices,
            secrets=resolve,
            tracer=make_tracer(settings),
        )
        graph = compile_graph(saver)
        for t in tasks:
            if browser is not None:
                browser.task_id = t.id
            task_secrets.clear()
            task_secrets.update(StaticSecretStore(t.secrets).secrets)
            run_id = f"eval-{t.id}-{tag}"
            limits = limits_from(settings, max_steps=t.max_steps)
            state = initial_state(
                ctx,
                task=t.task,
                allowed_domains=t.allowed_domains,
                run_id=run_id,
                start_url=t.start_url,
                limits=limits,
            )
            print(f"\n▶ {t.id}: {t.task}", flush=True)
            started = time.monotonic()
            try:
                final, _ = await drive(
                    graph,
                    ctx,
                    run_id,
                    state,
                    max_steps=limits["max_steps"],
                    on_event=print_event if verbose else None,
                    on_human=responder(t.human),
                    trace_tags=["eval", suite_name, t.id],
                )
                passed, why = check(t.expect, final)
            except Exception as err:  # one broken task must not end the suite
                final, passed, why = {}, False, f"crashed: {type(err).__name__}: {err}"[:300]
            r = TaskResult(
                id=t.id,
                passed=passed,
                why=why,
                status=str(final.get("status", "error")),
                steps=int(final.get("step", 0)),
                cost_usd=float(final.get("cost_usd", 0.0)),
                duration_s=round(time.monotonic() - started, 1),
                result=final.get("result"),
            )
            results.append(r)
            mark = "✔" if passed else "✖"
            print(
                f"{mark} {t.id}: {r.steps} steps · ${r.cost_usd:.4f} · {r.duration_s}s · {why}"
                f"\n  answer: {(r.result or '')[:200]}",
                flush=True,
            )
        if ctx.tracer is not None:
            ctx.tracer.flush()
    return results


def summarize(results: list[TaskResult]) -> dict[str, Any]:
    passed = [r for r in results if r.passed]
    durations = sorted(r.duration_s for r in results)
    p95 = durations[min(len(durations) - 1, int(0.95 * len(durations)))] if durations else 0.0
    return {
        "passed": len(passed),
        "total": len(results),
        "success_rate": round(len(passed) / len(results), 3) if results else 0.0,
        "median_steps": statistics.median(r.steps for r in results) if results else 0,
        "mean_cost_usd": round(statistics.fmean(r.cost_usd for r in results), 4)
        if results
        else 0.0,
        "total_cost_usd": round(sum(r.cost_usd for r in results), 4),
        "p95_duration_s": p95,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.runner")
    parser.add_argument("suite", type=Path)
    parser.add_argument("--only", action="append", help="task id to run; repeatable")
    parser.add_argument("--min-pass", type=int, default=0, help="exit 1 below this many passes")
    parser.add_argument(
        "--min-rate", type=float, default=0.0, help="exit 1 below this success rate"
    )
    parser.add_argument("--har", choices=["record", "replay", "replay-strict"])
    parser.add_argument("--har-dir", type=Path, default=Path("evals/har"))
    parser.add_argument("--summary-md", type=Path, help="append a Markdown summary here")
    parser.add_argument("--postgres", action="store_true", help="checkpoint to Postgres")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every step")
    parser.add_argument("--report-dir", type=Path, default=Path("evals/reports"))
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging("WARNING", json=False, redact=settings.redact_pii)
    if settings.llm_api_key is None:
        print(f"{settings.llm_key_name} is not set (put it in .env)", file=sys.stderr)
        return 2
    tasks = [t for t in load_suite(args.suite) if not args.only or t.id in args.only]

    results = asyncio.run(
        run_suite(
            tasks,
            settings,
            memory=not args.postgres,
            verbose=args.verbose,
            har=args.har,
            har_dir=args.har_dir,
            suite_name=args.suite.stem,
        )
    )
    summary = summarize(results)
    print(f"\n{json.dumps(summary, indent=2)}")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = args.report_dir / f"{args.suite.stem}-{stamp}.json"
    report.write_text(
        json.dumps(
            {
                "suite": str(args.suite),
                "created_at": stamp,
                "provider": settings.llm_provider,
                "models": {"decide": settings.model_decide, "small": settings.model_small},
                "prompt_version": prompt_version(),
                "summary": summary,
                "results": [asdict(r) for r in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"report: {report}")
    if args.summary_md:
        report_md.main([str(report), "--out", str(args.summary_md)])
    ok = summary["passed"] >= args.min_pass and summary["success_rate"] >= args.min_rate
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
