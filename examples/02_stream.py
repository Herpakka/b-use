"""Follow a run live over server-sent events, one line per step.

The stream survives reconnects: pass the last seen event id as ``after`` (the SDK and browsers
send it as ``Last-Event-ID``), even if the next request lands on another replica.
"""

import os

from bu_client import Client

with Client(os.environ.get("BU_URL", "http://localhost:8000"), os.environ["BU_API_KEY"]) as bu:
    run = bu.create_run(
        "Who wrote the first quote on the second page?",
        allowed_domains=["quotes.toscrape.com"],
        start_url="https://quotes.toscrape.com/",
    )
    for event in bu.stream(run["id"]):
        if event.type == "step":
            print(f"  {event.data['line']}")
        elif event.type == "decision":
            print(f"→ {event.data['tool']} {event.data['args']}")
        elif event.type == "finished":
            print(f"{event.data['status']}: {event.data['result'] or event.data['failure_reason']}")
            break
