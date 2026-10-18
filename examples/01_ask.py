"""Ask a question about a website and print the answer.

BU_API_KEY=... python examples/01_ask.py
"""

import os

from bu_client import Client

with Client(os.environ.get("BU_URL", "http://localhost:8000"), os.environ["BU_API_KEY"]) as bu:
    run = bu.create_run(
        "What is the price of the book 'A Light in the Attic'?",
        allowed_domains=["books.toscrape.com"],
        start_url="https://books.toscrape.com/",
    )
    final = bu.wait(run["id"])
    print(final["status"], "→", final["result"] or final["failure_reason"])
    print(f"{final['steps']} steps, ${final['cost_usd']:.4f} {final['cost_breakdown']}")
