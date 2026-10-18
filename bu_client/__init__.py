"""Thin Python client for the Browser Use Agent API.

from bu_client import Client

client = Client("http://localhost:8000", api_key="…")
run = client.create_run("Price of 'A Light in the Attic'?", ["books.toscrape.com"])
final = client.wait(run["id"], on_event=print, on_human=lambda p: {"approved": True})
print(final["result"])
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx

__all__ = ["ApiError", "Client", "Event"]

TERMINAL = frozenset({"done", "failed", "cancelled"})


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: Any) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class Event:
    id: int
    type: str
    data: dict[str, Any]


def parse_sse(lines: Iterator[str]) -> Iterator[Event]:
    """Minimal SSE parser: id/event/data fields; comment lines (heartbeats) are skipped."""
    fields: dict[str, str] = {}
    for line in lines:
        if line == "":
            if "event" in fields:
                yield Event(
                    id=int(fields.get("id", "0")),
                    type=fields["event"],
                    data=json.loads(fields.get("data", "{}")),
                )
            fields = {}
        elif not line.startswith(":") and ":" in line:
            name, _, value = line.partition(":")
            fields[name] = value[1:] if value.startswith(" ") else value


class Client:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "",
        *,
        timeout: float = 30.0,
        http: httpx.Client | None = None,
    ) -> None:
        self._http = http or httpx.Client(base_url=base_url, timeout=timeout)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        res = self._http.request(method, path, headers=self._headers, **kwargs)
        if res.status_code >= 400:
            try:
                detail = res.json().get("detail", res.text)
            except ValueError:
                detail = res.text
            raise ApiError(res.status_code, detail)
        return res

    # ------------------------------------------------------------------ runs
    def create_run(
        self,
        task: str,
        allowed_domains: list[str],
        *,
        start_url: str | None = None,
        session: str | None = None,
        vision: bool = False,
        max_steps: int | None = None,
        budget_usd: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"task": task, "allowed_domains": allowed_domains, "vision": vision}
        for key, value in (
            ("start_url", start_url),
            ("session", session),
            ("max_steps", max_steps),
            ("budget_usd", budget_usd),
        ):
            if value is not None:
                body[key] = value
        result: dict[str, Any] = self._request("POST", "/v1/runs", json=body).json()
        return result

    def get_run(self, run_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._request("GET", f"/v1/runs/{run_id}").json()
        return result

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._request(
            "GET", "/v1/runs", params={"limit": limit}
        ).json()
        return result

    def events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._request(
            "GET", f"/v1/runs/{run_id}/events", params={"after": after}
        ).json()
        return result

    def resume(self, run_id: str, *, approved: bool = True, answer: str = "") -> dict[str, Any]:
        result: dict[str, Any] = self._request(
            "POST", f"/v1/runs/{run_id}/resume", json={"approved": approved, "answer": answer}
        ).json()
        return result

    def cancel(self, run_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._request("POST", f"/v1/runs/{run_id}/cancel").json()
        return result

    def screenshot(self, run_id: str) -> bytes:
        return self._request("GET", f"/v1/runs/{run_id}/screenshot").content

    def stream(self, run_id: str, after: int = 0) -> Iterator[Event]:
        """Events from ``after`` on, live; ends after the ``finished`` event."""
        headers = {**self._headers, "Accept": "text/event-stream"}
        with self._http.stream(
            "GET",
            f"/v1/runs/{run_id}/stream",
            params={"after": after},
            headers=headers,
            timeout=httpx.Timeout(None, connect=10),
        ) as res:
            if res.status_code >= 400:
                res.read()
                raise ApiError(res.status_code, res.text)
            for event in parse_sse(res.iter_lines()):
                yield event
                if event.type == "finished":
                    return

    def wait(
        self,
        run_id: str,
        *,
        on_event: Callable[[Event], None] | None = None,
        on_human: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Follow a run to the end. ``on_human(pending) -> {"approved", "answer"}`` answers
        approvals/questions; without it, a run that needs a human is returned as it is."""
        last = 0
        while True:
            for event in self.stream(run_id, after=last):
                last = event.id
                if on_event is not None:
                    on_event(event)
                if event.type == "needs_human":
                    if on_human is None:
                        return self.get_run(run_id)
                    # The event log is PII-redacted; the run record has the question as asked.
                    pending = self.get_run(run_id).get("pending") or event.data
                    self.resume(run_id, **on_human(pending))
            run = self.get_run(run_id)
            if run["status"] in TERMINAL:
                return run
