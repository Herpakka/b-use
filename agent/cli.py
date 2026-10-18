"""``bu`` — run browser tasks from the terminal.

    bu run "<task>" --allow books.toscrape.com [--start-url URL] [--session NAME] [--vision]
    bu resume <run_id>

Human-in-the-loop prompts (approvals, questions) are answered on stdin. Checkpoints go to
Postgres (BU_DATABASE_URL) so a run survives a crash and continues with ``bu resume``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from agent.context import AgentContext
from agent.graph import compile_graph
from agent.llm import make_chat_model
from agent.runner import drive, initial_state, limits_from, open_checkpointer
from browser.local import local_browser
from core.config import Settings, get_settings
from core.costs import PriceTable
from core.logging import bind_context, configure_logging
from core.secrets import StaticSecretStore


def _print(text: str = "") -> None:
    print(text, flush=True)


def print_event(node: str, update: dict[str, Any]) -> None:
    if node == "plan":
        _print("plan:")
        for i, goal in enumerate(update.get("plan", []), 1):
            _print(f"  {i}. {goal}")
    elif node == "observe":
        if url := update.get("current_url"):
            _print(f"  · {url}")
    elif node == "decide":
        if update.get("status") == "failed":
            _print(f"✖ {update.get('failure_reason')}")
        elif pending := update.get("pending"):
            args = json.dumps(pending["args"], ensure_ascii=False)
            _print(f"→ {pending['name']} {args[:160]}")
            if update.get("status") == "needs_human":
                _print(f"  ⏸ needs human: {update.get('pending_reason')}")
    elif node in ("act", "reflect", "human"):
        for line in update.get("history", []):
            _print(f"  {line}")
        if node == "reflect" and update.get("notes"):
            _print(f"  advice: {update['notes']}")
        last = update.get("last_result") or {}
        if node == "act" and last.get("stale"):
            _print(f"  ↻ {last.get('message')}")


async def ask_stdin(payload: dict[str, Any]) -> dict[str, Any]:
    _print()
    if payload.get("kind") == "question":
        _print(f"🙋 The agent asks: {payload.get('question')}")
        answer = await asyncio.to_thread(input, "your answer> ")
        return {"approved": True, "answer": answer}
    _print(f"🛑 Approval needed: {payload.get('action')}")
    _print(f"   reason: {payload.get('reason')}   page: {payload.get('url')}")
    reply = (await asyncio.to_thread(input, "approve? [y/N] ")).strip().lower()
    return {"approved": reply in ("y", "yes"), "answer": ""}


def print_summary(state: dict[str, Any], run_id: str) -> None:
    _print()
    status = state.get("status")
    if status == "done":
        _print(f"✔ done: {state.get('result')}")
    else:
        _print(f"✖ {status}: {state.get('failure_reason')}")
        if state.get("result"):
            _print(f"  result: {state.get('result')}")
    _print(
        f"  run {run_id} · {state.get('step', 0)} steps · ${state.get('cost_usd', 0):.4f}"
        f" · {state.get('tokens_in', 0)} in / {state.get('tokens_out', 0)} out tokens"
    )


async def _main(args: argparse.Namespace, settings: Settings) -> int:
    prices = PriceTable.from_settings(settings)
    prices.require(settings.model_decide, settings.model_small)
    async with local_browser(settings) as browser:
        ctx = AgentContext(
            browser=browser,
            llm_decide=make_chat_model(settings, settings.model_decide),
            llm_small=make_chat_model(settings, settings.model_small),
            settings=settings,
            prices=prices,
            secrets=StaticSecretStore(settings.local_secrets).get,
        )
        async with open_checkpointer(settings, memory=args.memory) as saver:
            graph = compile_graph(saver)
            if args.cmd == "run":
                limits = limits_from(
                    settings,
                    max_steps=args.max_steps,
                    budget_usd=args.budget,
                    run_timeout_s=args.timeout,
                )
                state = initial_state(
                    ctx,
                    task=args.task,
                    allowed_domains=args.allow,
                    run_id=args.run_id,
                    session_id=args.session,
                    start_url=args.start_url,
                    vision=args.vision,
                    limits=limits,
                )
                run_id = state["run_id"]
                payload: Any = state
                _print(f"run {run_id}  (resume with: bu resume {run_id})")
            else:
                run_id = args.run_id
                snapshot = await graph.aget_state({"configurable": {"thread_id": run_id}})
                if not snapshot.values:
                    _print(f"unknown run {run_id}")
                    return 2
                if not snapshot.next:
                    print_summary(dict(snapshot.values), run_id)
                    return 0 if snapshot.values.get("status") == "done" else 1
                limits = snapshot.values["limits"]
                payload = None
                _print(f"resuming {run_id} at {', '.join(snapshot.next)}")
            bind_context(run_id=run_id)
            final, _ = await drive(
                graph,
                ctx,
                run_id,
                payload,
                max_steps=limits["max_steps"],
                on_event=print_event,
                on_human=ask_stdin,
            )
    print_summary(final, run_id)
    return 0 if final.get("status") == "done" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bu", description="LLM-driven browser tasks.")
    parser.add_argument("-v", "--verbose", action="store_true", help="show structured logs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="start a new run")
    run.add_argument("task")
    run.add_argument(
        "--allow",
        action="append",
        required=True,
        metavar="DOMAIN",
        help="allowed domain (subdomains included); repeat for more",
    )
    run.add_argument("--start-url")
    run.add_argument("--session", help="session id to reuse cookies/logins across runs")
    run.add_argument("--run-id")
    run.add_argument("--vision", action="store_true", help="send a screenshot every step")
    run.add_argument("--max-steps", type=int)
    run.add_argument("--budget", type=float, help="USD budget for this run")
    run.add_argument("--timeout", type=float, help="wall-clock seconds")
    run.add_argument("--memory", action="store_true", help="in-memory checkpoints (no resume)")

    resume = sub.add_parser("resume", help="continue a run from its last checkpoint")
    resume.add_argument("run_id")
    resume.set_defaults(memory=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(
        "DEBUG" if args.verbose else "WARNING", json=False, redact=settings.redact_pii
    )
    if settings.openai_api_key is None:
        print("BU_OPENAI_API_KEY is not set (put it in .env)", file=sys.stderr)
        return 2
    try:
        return asyncio.run(_main(args, settings))
    except KeyboardInterrupt:
        print("\ninterrupted — continue later with: bu resume <run_id>", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
