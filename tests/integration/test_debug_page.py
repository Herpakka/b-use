"""The debug page's JavaScript, driven in a real Chromium against a real server."""

from __future__ import annotations

import re

import pytest
from playwright.async_api import async_playwright, expect

from tests.agent_fakes import PLAN, FakeBrowser, tool
from tests.api_support import HOME, KEY_A, SHOP, live_server, memory_factory

pytestmark = pytest.mark.browser


async def test_start_run_approve_and_see_result() -> None:
    script = [PLAN, tool("Click", index=2), tool("Done", result="Ordered 1 kettle")]
    browser_fake = FakeBrowser(SHOP)
    with live_server(memory_factory(browser_fake, script)) as base_url:
        async with async_playwright() as pw:
            try:
                chromium = await pw.chromium.launch()
            except Exception as err:
                pytest.skip(f"Chromium unavailable: {str(err).splitlines()[0][:120]}")
            page = await chromium.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda exc: errors.append(str(exc)))

            await page.goto(f"{base_url}/debug")
            await page.fill("#key", KEY_A)
            await page.click("#save-key")
            await expect(page.locator("#auth-state")).to_have_text("✓ connected")

            await page.fill("#task", "Order a kettle")
            await page.fill("#domains", "shop.test")
            await page.fill("#start", HOME)
            await page.click("#new-run button[type=submit]")

            human = page.locator("#human")
            await expect(human).to_be_visible(timeout=10_000)
            await expect(page.locator("#h-title")).to_have_text("Approval needed")
            await expect(page.locator("#h-body")).to_contain_text('Click [2] "Place order"')
            await expect(page.locator("#r-status")).to_have_text("needs_human")

            await page.click("#h-approve")
            await expect(page.locator("#r-status")).to_have_text("done", timeout=10_000)
            await expect(page.locator("#r-result")).to_have_text("Ordered 1 kettle")
            await expect(human).to_be_hidden()

            log = page.locator("#log")
            await expect(log).to_contain_text("needs_human")
            await expect(log).to_contain_text("human approved")
            await expect(log).to_contain_text("finished")
            # the screenshot is fetched with the key and shown from a blob: URL
            await expect(page.locator("#shot")).to_have_attribute("src", re.compile(r"^blob:"))
            await expect(page.locator("#runs li")).to_have_count(1)
            assert errors == []
            await chromium.close()
