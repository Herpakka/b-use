"""Secrets over the API: stored per tenant, never echoed, checked when a run is created."""

from typing import Any

from browser.models import ActionResult
from core.secrets import MemorySecretStore, Secret
from tests.agent_fakes import PLAN, FakeBrowser, page, text, tool
from tests.api_support import (
    AUTH_A,
    AUTH_B,
    HOME,
    client_for,
    memory_factory,
    start_run,
    status_in,
    wait_for,
)

VALUE = "correct-horse-battery"
LOGIN = {HOME: page(HOME, ("input", "Password", "password"), ("button", "Sign in"))}


def test_store_list_and_delete_never_show_values() -> None:
    factory = memory_factory(FakeBrowser(), [])
    with client_for(factory) as client:
        res = client.put(
            "/v1/secrets/shop_pw", json={"value": VALUE, "domains": ["Shop.Test"]}, headers=AUTH_A
        )
        assert res.status_code == 204, res.text
        listed = client.get("/v1/secrets", headers=AUTH_A)
        assert [(s["name"], s["domains"]) for s in listed.json()] == [("shop_pw", ["shop.test"])]
        assert VALUE not in listed.text
        assert client.get("/v1/secrets", headers=AUTH_B).json() == []  # per tenant

        short = client.put("/v1/secrets/x", json={"value": "abc"}, headers=AUTH_A)
        assert short.status_code == 422 and "abc" not in short.text
        bad_name = client.put("/v1/secrets/a b", json={"value": VALUE}, headers=AUTH_A)
        assert bad_name.status_code == 422 and VALUE not in bad_name.text

        assert client.delete("/v1/secrets/shop_pw", headers=AUTH_A).status_code == 204
        assert client.delete("/v1/secrets/shop_pw", headers=AUTH_A).status_code == 404


def test_runs_check_their_placeholders_and_use_the_tenant_scope() -> None:
    secrets = MemorySecretStore()
    browser = FakeBrowser(LOGIN, results=[ActionResult(ok=True, message="typed → ok")])
    script = [PLAN, tool("TypeText", index=1, text="{{SECRET:pw}}"), tool("Done", result="in")]
    factory = memory_factory(browser, script, [text("unused")], secrets=secrets)
    with client_for(factory) as client:
        body: dict[str, Any] = {
            "task": "log in with {{SECRET:pw}}",
            "allowed_domains": ["shop.test"],
            "start_url": HOME,
        }
        res = client.post("/v1/runs", json=body, headers=AUTH_A)
        assert res.status_code == 422 and "unknown secret 'pw'" in res.text

        client.put("/v1/secrets/pw", json={"value": VALUE}, headers=AUTH_A)
        run = start_run(client, **body)
        final = wait_for(client, run["id"], status_in("done", "failed"))
        assert final["status"] == "done", final
        assert browser.secrets[1] == {"pw": Secret("pw", VALUE)}
        events = client.get(f"/v1/runs/{run['id']}/events", headers=AUTH_A).text
        assert VALUE not in events and "{{SECRET:pw}}" in events
        # Another tenant can't use acme's secret.
        res = client.post("/v1/runs", json=body, headers=AUTH_B)
        assert res.status_code == 422
