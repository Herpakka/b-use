"""The agent hands secret values to the browser and nowhere else."""

from browser.actions import TypeText
from browser.models import ActionResult
from core.secrets import MemorySecretStore, Secret
from tests.agent_fakes import PLAN, FakeBrowser, decide_prompts, make_ctx, page, run_task, tool

HOME = "https://shop.test/login"
LOGIN = {HOME: page(HOME, ("input", "Password", "password"), ("button", "Sign in"))}
VALUE = "correct-horse-battery"
# A refused step is an error, so the run reflects before deciding again.
REFLECT = tool(
    "Reflection", diagnosis="the secret is unavailable", revised_plan=["stop"], advice=""
)


async def store() -> MemorySecretStore:
    s = MemorySecretStore()
    await s.put("acme", "pw", VALUE, ["shop.test"])
    return s


async def test_values_go_to_the_browser_only() -> None:
    secrets = await store()
    # A misbehaving browser layer that echoes the value: the agent masks it anyway.
    browser = FakeBrowser(LOGIN, results=[ActionResult(ok=True, message=f"typed {VALUE} → ok")])
    script = [PLAN, tool("TypeText", index=1, text="{{SECRET:pw}}"), tool("Done", result="in")]
    ctx = make_ctx(browser, script, secrets=secrets.get)
    final, _, _ = await run_task(
        ctx, task="Log in with {{SECRET:pw}}", start_url=HOME, secret_scope="acme"
    )

    assert final["status"] == "done"
    assert browser.executed[1] == TypeText(index=1, text="{{SECRET:pw}}")
    assert browser.secrets[1] == {"pw": Secret("pw", VALUE, ("shop.test",))}
    assert browser.secrets[0] == {}  # nothing for the start navigation
    assert final["history"][0] == "1. typed {{SECRET:pw}} → ok"
    assert all(VALUE not in p for p in decide_prompts(ctx))
    assert VALUE not in repr(final)


async def test_other_scopes_and_missing_names_are_refused() -> None:
    secrets = await store()
    browser = FakeBrowser(LOGIN)
    script = [
        PLAN,
        tool("TypeText", index=1, text="{{SECRET:pw}}"),
        REFLECT,
        tool("Done", result="gave up", success=False),
    ]
    ctx = make_ctx(browser, script, secrets=secrets.get)
    final, _, _ = await run_task(ctx, start_url=HOME, secret_scope="globex")

    assert final["history"][0] == "1. TypeText → refused: no secret named 'pw' is stored"
    assert len(browser.executed) == 1  # only the start navigation reached the browser


async def test_without_a_secret_store() -> None:
    browser = FakeBrowser(LOGIN)
    script = [
        PLAN,
        tool("TypeText", index=1, text="{{SECRET:pw}}"),
        REFLECT,
        tool("Done", result="x"),
    ]
    final, _, _ = await run_task(make_ctx(browser, script), start_url=HOME)
    assert "secrets are not configured" in final["history"][0]
