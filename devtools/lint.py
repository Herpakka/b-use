"""``uv run lint``: ruff → ruff format → mypy (strict) → import-linter layer contracts."""

from __future__ import annotations

import subprocess
import sys

STEPS: list[list[str]] = [
    ["ruff", "check", "."],
    ["ruff", "format", "--check", "."],
    ["mypy"],
    ["lint-imports"],
]


def main() -> int:
    failed: list[str] = []
    for cmd in STEPS:
        print(f"$ {' '.join(cmd)}", flush=True)
        if subprocess.run(cmd, check=False).returncode != 0:
            failed.append(" ".join(cmd))
    if failed:
        print(f"\nlint failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nlint ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
