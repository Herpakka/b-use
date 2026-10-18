"""Versioned prompt files. Bump the ``<!-- version: N -->`` header whenever a prompt changes;
the combined version string is stored on every run so eval results can be compared."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

PROMPT_NAMES = ("system", "plan", "reflect", "extract")
_VERSION = re.compile(r"^<!--\s*version:\s*(\S+)\s*-->\s*\n")


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    text: str


@lru_cache
def load(name: str) -> Prompt:
    raw = files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")
    match = _VERSION.match(raw)
    if match is None:
        raise ValueError(f"prompt {name}.md is missing its '<!-- version: N -->' header")
    return Prompt(name=name, version=match.group(1), text=raw[match.end() :].strip())


def prompt_version() -> str:
    return ",".join(f"{n}@{load(n).version}" for n in PROMPT_NAMES)
