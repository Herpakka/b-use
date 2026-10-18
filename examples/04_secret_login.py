"""Log in with a stored credential the model never sees.

The password is stored once (encrypted; the server needs BU_SECRETS_KEY) and bound to the
login domain. The task refers to it as {{SECRET:demo_pw}}; the browser layer types it there and
nowhere else, and the agent's prompts and the run's events only ever contain the placeholder.
"""

import os

from bu_client import Client

with Client(os.environ.get("BU_URL", "http://localhost:8000"), os.environ["BU_API_KEY"]) as bu:
    bu.put_secret("demo_pw", "SuperSecretPassword!", domains=["the-internet.herokuapp.com"])
    run = bu.create_run(
        "Log in with the username 'tomsmith' and the password {{SECRET:demo_pw}}, then report"
        " the message shown after login.",
        allowed_domains=["the-internet.herokuapp.com"],
        start_url="https://the-internet.herokuapp.com/login",
    )
    final = bu.wait(run["id"])
    print(final["status"], "→", final["result"] or final["failure_reason"])
    print("stored secrets:", [s["name"] for s in bu.list_secrets()])
