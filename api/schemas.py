"""Request/response models of the public HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator

from core.store import RunEvent, RunRecord, RunStatus

SESSION_NAME = r"^[A-Za-z0-9_-]{1,40}$"


class CreateRunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=4000)
    allowed_domains: list[str] = Field(
        min_length=1, max_length=50, description="Navigation is refused outside these domains."
    )
    start_url: str | None = None
    session: str | None = Field(
        default=None,
        pattern=SESSION_NAME,
        description="Reuse cookies/logins across runs under this name (per tenant).",
    )
    vision: bool = False
    max_steps: int | None = Field(default=None, ge=1, le=100)
    budget_usd: float | None = Field(default=None, gt=0)

    @field_validator("start_url")
    @classmethod
    def _http_only(cls, v: str | None) -> str | None:
        if v is not None and urlsplit(v).scheme not in ("http", "https"):
            raise ValueError("start_url must be http(s)")
        return v

    @field_validator("allowed_domains")
    @classmethod
    def _non_blank(cls, v: list[str]) -> list[str]:
        cleaned = [d.strip() for d in v if d.strip()]
        if not cleaned:
            raise ValueError("at least one domain is required")
        return cleaned


class ResumeRequest(BaseModel):
    approved: bool = True
    answer: str = Field(default="", max_length=2000)


class RunOut(BaseModel):
    id: str
    status: RunStatus
    task: str
    allowed_domains: list[str]
    start_url: str | None
    session: str
    result: str | None
    failure_reason: str | None
    steps: int
    cost_usd: float
    tokens_in: int
    tokens_out: int
    pending: dict[str, Any] | None = Field(
        description="What the agent is waiting for while status is needs_human."
    )
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None

    @classmethod
    def of(cls, run: RunRecord) -> RunOut:
        return cls(
            **run.model_dump(exclude={"tenant", "session_id", "max_steps", "prompt_version"}),
            session=session_name(run.tenant, run.session_id),
        )


class EventOut(BaseModel):
    seq: int
    type: str
    data: dict[str, Any]
    created_at: datetime

    @classmethod
    def of(cls, event: RunEvent) -> EventOut:
        return cls(seq=event.seq, type=event.type, data=event.data, created_at=event.created_at)


class SessionOut(BaseModel):
    name: str
    url: str
    tabs: int
    idle_s: float
    busy: bool
    held: bool = Field(default=False, description="Kept open because its run waits for a human.")


def session_id(tenant: str, name: str) -> str:
    """Browser session ids are namespaced per tenant so cookies never cross tenants."""
    return f"{tenant}.{name}"


def session_name(tenant: str, sid: str) -> str:
    prefix = f"{tenant}."
    return sid[len(prefix) :] if sid.startswith(prefix) else sid
