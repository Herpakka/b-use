"""Reuse a login across runs with a named session, and keep an eye on spend.

The first run logs in under session "quotes-demo"; its cookies are saved when it ends. The
second run, under the same name, starts already logged in.
"""

import os

from bu_client import Client

SESSION = "quotes-demo"
DOMAINS = ["quotes.toscrape.com"]

with Client(os.environ.get("BU_URL", "http://localhost:8000"), os.environ["BU_API_KEY"]) as bu:
    first = bu.create_run(
        "Log in with username 'demo' and password 'demo' (any password works), then report the"
        " link shown at the top right.",
        DOMAINS,
        start_url="https://quotes.toscrape.com/login",
        session=SESSION,
    )
    print("1:", bu.wait(first["id"])["result"])

    second = bu.create_run(
        "Open the home page and report whether you are logged in (is there a Logout link?).",
        DOMAINS,
        start_url="https://quotes.toscrape.com/",
        session=SESSION,
    )
    print("2:", bu.wait(second["id"])["result"])

    usage = bu.usage()
    print(
        f"{usage['month']}: ${usage['cost_usd']:.4f} over {usage['runs']} runs;"
        f" left: {usage['remaining_usd']}"
    )
