"""End-to-end check of the staging stack (docker/compose.staging.yml): real runs through the
proxy while replicas are killed and stopped. Costs a few cents of OpenAI.

    uv run python -m devtools.staging_check [--url http://localhost:8088]

1. failover   — SIGKILL the replica that owns a running run: the other adopts and finishes it.
2. drain      — SIGTERM the owner mid-run: it lets the run finish before exiting.
3. forwarding — a run paused for approval is resumed through the proxy, whichever replica
                answers; the owner (which holds the paused browser) carries it on.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from typing import Any

import httpx

from bu_client import Client
from core.config import get_settings

PROJECT = "browser-use-staging"
BOOKS = ["books.toscrape.com"]
TASK = "What is the price of the book 'A Light in the Attic'? Check its detail page."


def docker(*args: str) -> str:
    return subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def container_of(replica: str) -> str:
    """The container whose hostname (= replica id) is ``replica``."""
    for name in docker(
        "ps", "-a", "--filter", f"name={PROJECT}-app", "--format", "{{.Names}}"
    ).split():
        if docker("inspect", "--format", "{{.Config.Hostname}}", name) == replica:
            return name
    raise SystemExit(f"no container for replica {replica}")


def wait(client: Client, run_id: str, predicate: Any, timeout: float, what: str) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            run = client.get_run(run_id)
            if predicate(run):
                return run
        except Exception:  # the proxy may answer 502 while a replica goes away
            pass
        time.sleep(0.5)
    raise SystemExit(f"timed out waiting for {what}")


def running_with_owner(client: Client, run_id: str) -> dict[str, Any]:
    return wait(
        client,
        run_id,
        lambda r: r["status"] == "running" and r["owner"] and r["steps"] >= 1,
        120,
        "the run to take its first step",
    )


def ensure_up(client: Client) -> None:
    docker("compose", "-f", "docker/compose.staging.yml", "up", "-d", "--no-build", "app")
    deadline = time.monotonic() + 120
    seen: set[str] = set()
    while time.monotonic() < deadline and len(seen) < 2:
        try:
            # No cookie jar: the proxy's sticky cookie would pin every probe to one replica.
            base = str(client._http.base_url).rstrip("/")
            ready = httpx.get(f"{base}/readyz", timeout=5).json()
            if ready["status"] == "ready":
                seen.add(ready["checks"]["replica"])
        except Exception:
            pass
        time.sleep(0.5)
    if len(seen) < 2:
        raise SystemExit(f"both replicas should be ready, saw {seen}")


def failover(client: Client) -> str:
    run = client.create_run(TASK, BOOKS, start_url="https://books.toscrape.com/")
    owner = running_with_owner(client, run["id"])["owner"]
    victim = container_of(owner)
    docker("kill", victim)  # no drain, no lease release: its leases just run out
    final = wait(client, run["id"], lambda r: r["status"] in ("done", "failed"), 180, "adoption")
    recovered = [e for e in client.events(run["id"]) if e["type"] == "recovered"]
    assert final["status"] == "done", final
    assert "51.77" in (final["result"] or ""), final["result"]
    assert recovered and recovered[0]["data"]["by"] != owner, recovered
    docker("start", victim)
    return f"run {run['id']}: owner {owner} killed, finished by {recovered[0]['data']['by']}"


def drain(client: Client) -> str:
    run = client.create_run(TASK, BOOKS, start_url="https://books.toscrape.com/")
    owner = running_with_owner(client, run["id"])["owner"]
    victim = container_of(owner)
    started = time.monotonic()
    stopper = threading.Thread(target=docker, args=("stop", "-t", "90", victim))
    stopper.start()
    final = wait(client, run["id"], lambda r: r["status"] in ("done", "failed"), 120, "draining")
    stopper.join()
    assert final["status"] == "done", final
    events = [e["type"] for e in client.events(run["id"])]
    assert "recovered" not in events, "a drained run should finish where it ran"
    docker("start", victim)
    return f"run {run['id']}: {owner} drained it in {time.monotonic() - started:.0f}s, then exited"


def forwarding(client: Client, key: str) -> str:
    run = client.create_run(
        "On the Add/Remove Elements page, add one element, then delete it and report what"
        " the page shows.",
        ["the-internet.herokuapp.com"],
        start_url="https://the-internet.herokuapp.com/add_remove_elements/",
    )
    paused = wait(client, run["id"], lambda r: r["status"] == "needs_human", 120, "the approval")
    owner = paused["owner"]
    names = docker("ps", "--filter", f"name={PROJECT}-app", "--format", "{{.Names}}").split()
    other = next(
        n for n in names if docker("inspect", "--format", "{{.Config.Hostname}}", n) != owner
    )
    # Straight to the replica that does NOT own the run (from inside the proxy's network).
    docker(
        "exec",
        f"{PROJECT}-proxy-1",
        "wget",
        "-qO-",
        "--header",
        f"Authorization: Bearer {key}",
        "--header",
        "Content-Type: application/json",
        "--post-data",
        '{"approved": true}',
        f"http://{other}:8000/v1/runs/{run['id']}/resume",
    )
    final = wait(client, run["id"], lambda r: r["status"] in ("done", "failed"), 120, "resume")
    assert final["status"] == "done", final
    recovered = [e for e in client.events(run["id"]) if e["type"] == "recovered"]
    assert not recovered, "the owner should have resumed it, not another replica"
    return f"run {run['id']}: paused on {owner}, resumed on {other}, carried on by {owner}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="devtools.staging_check")
    parser.add_argument("--url", default="http://localhost:8088")
    parser.add_argument("--only", choices=["failover", "drain", "forwarding"], action="append")
    args = parser.parse_args(argv)
    keys = get_settings().api_keys
    if not keys:
        print("BU_API_KEYS is empty (.env)", file=sys.stderr)
        return 2
    checks: dict[str, Any] = {"failover": failover, "drain": drain, "forwarding": forwarding}
    results: dict[str, str] = {}
    key = next(iter(keys))
    with Client(args.url, key, timeout=60) as client:
        for name, check in checks.items():
            if args.only and name not in args.only:
                continue
            ensure_up(client)
            print(f"▶ {name} …", flush=True)
            results[name] = check(client, key) if name == "forwarding" else check(client)
            print(f"✔ {name}: {results[name]}", flush=True)
    print(json.dumps({"passed": list(results)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
