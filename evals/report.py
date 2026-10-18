"""Compare an eval report with the previous one and render Markdown (job summary, PR comment).

uv run python -m evals.report evals/reports/tasks-<stamp>.json [--previous other.json]
Without ``--previous`` the latest earlier report of the same suite in the same folder is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Plan §5 targets: success rate (MVP / v1), cost per task, median steps, p95 duration.
SUCCESS_MVP, SUCCESS_V1 = 0.55, 0.70
TARGETS: dict[str, float] = {"mean_cost_usd": 0.15, "median_steps": 9, "p95": 150}


def previous_report(current: Path) -> Path | None:
    suite = current.name.rsplit("-", 1)[0]
    earlier = sorted(p for p in current.parent.glob(f"{suite}-*.json") if p.name < current.name)
    return earlier[-1] if earlier else None


def _delta(now: float, before: float | None, fmt: str) -> str:
    if before is None:
        return ""
    d = now - before
    return "" if abs(d) < 1e-9 else f" ({'+' if d > 0 else ''}{fmt.format(d)})"


def render(current: dict[str, Any], previous: dict[str, Any] | None = None) -> str:
    s, p = current["summary"], (previous or {}).get("summary", {})
    before = {r["id"]: r for r in (previous or {}).get("results", [])}
    mvp, v1 = SUCCESS_MVP, SUCCESS_V1
    rate = s["success_rate"]
    verdict = "✅ v1 target" if rate >= v1 else "🟡 MVP target" if rate >= mvp else "❌ below MVP"

    def row(label: str, key: str, fmt: str, delta_fmt: str, target: str, scale: float = 1) -> str:
        now = float(s[key]) * scale
        before = float(p[key]) * scale if key in p else None
        return f"| {label} | {fmt.format(now)}{_delta(now, before, delta_fmt)} | {target} |"

    lines = [
        f"## Eval `{Path(current['suite']).name}` — {s['passed']}/{s['total']} passed {verdict}",
        "",
        f"Models `{current['models']['decide']}` / `{current['models']['small']}` ·"
        f" prompts `{current['prompt_version']}` · {current['created_at']}",
        "",
        "| | now | target |",
        "|---|---|---|",
        row(
            "success rate",
            "success_rate",
            "{:.0f} %",
            "{:.0f} pt",
            f"≥ {mvp:.0%} MVP · ≥ {v1:.0%} v1",
            100,
        ),
        row(
            "mean cost / task",
            "mean_cost_usd",
            "${:.4f}",
            "{:.4f}",
            f"≤ ${TARGETS['mean_cost_usd']:.2f}",
        ),
        row("median steps", "median_steps", "{:g}", "{:g}", f"≤ {TARGETS['median_steps']:g}"),
        row("p95 duration", "p95_duration_s", "{:.1f} s", "{:.1f} s", f"≤ {TARGETS['p95']:g} s"),
        f"| total cost | ${s['total_cost_usd']:.4f} | |",
        "",
    ]
    regressions = [
        r for r in current["results"] if not r["passed"] and before.get(r["id"], {}).get("passed")
    ]
    fixed = [
        r
        for r in current["results"]
        if r["passed"] and before.get(r["id"], {}).get("passed") is False
    ]
    if regressions:
        lines += ["**Regressions** (passed last time):", ""]
        lines += [f"- `{r['id']}` — {r['why']}" for r in regressions] + [""]
    if fixed:
        lines += ["**Fixed since last time:** " + ", ".join(f"`{r['id']}`" for r in fixed), ""]
    lines += ["| task | | steps | cost | time | why |", "|---|---|---|---|---|---|"]
    for r in current["results"]:
        mark = "✔" if r["passed"] else "✖"
        why = "" if r["passed"] else str(r["why"]).replace("|", "\\|")[:120]
        cost = f"${r['cost_usd']:.4f}"
        lines.append(
            f"| `{r['id']}` | {mark} | {r['steps']} | {cost} | {r['duration_s']} s | {why} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.report")
    parser.add_argument("report", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--out", type=Path, help="write the Markdown here (default: stdout)")
    args = parser.parse_args(argv)
    current = json.loads(args.report.read_text(encoding="utf-8"))
    prev_path = args.previous or previous_report(args.report)
    previous = (
        json.loads(prev_path.read_text(encoding="utf-8"))
        if prev_path is not None and prev_path.exists()
        else None
    )
    text = render(current, previous)
    if args.out:
        with args.out.open("a", encoding="utf-8") as fh:  # $GITHUB_STEP_SUMMARY is appended to
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
