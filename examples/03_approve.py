"""Answer the agent's approvals and questions from the terminal.

Clicks that pay, order, send, delete or submit a form always pause for a human. Here the
agent is asked to log in, which it may do; anything with lasting effects is put to you.
"""

import os
from typing import Any

from bu_client import Client


def ask(pending: dict[str, Any]) -> dict[str, Any]:
    if pending["kind"] == "question":
        return {"approved": True, "answer": input(f"agent asks: {pending['question']}\n> ")}
    print(f"approval needed: {pending['action']}\n  because {pending['reason']} ({pending['url']})")
    return {"approved": input("approve? [y/N] ").strip().lower() == "y"}


with Client(os.environ.get("BU_URL", "http://localhost:8000"), os.environ["BU_API_KEY"]) as bu:
    run = bu.create_run(
        "On the-internet.herokuapp.com, open the Add/Remove Elements page, add one element,"
        " then delete it and report what the page shows.",
        allowed_domains=["the-internet.herokuapp.com"],
        start_url="https://the-internet.herokuapp.com/add_remove_elements/",
    )
    final = bu.wait(run["id"], on_human=ask)
    print(final["status"], "→", final["result"] or final["failure_reason"])
