"""5.2: a HAR recorded from a live site serves the same pages later, without the network."""

from __future__ import annotations

from pathlib import Path

import pytest

from browser.actions import Navigate
from browser.egress import EgressProxy
from browser.pool import BrowserPool
from evals.har import HarBrowserService
from tests.adversarial.web import serve_web
from tests.conftest import make_settings

pytestmark = pytest.mark.browser


async def test_record_then_replay_pins_the_site(tmp_path: Path) -> None:
    with serve_web() as web:
        url = web.page("/price", "<h1>Blue Kettle</h1><p>Price: ฿1,290</p>")
        pool = BrowserPool(
            make_settings(tmp_path / "s", block_resources=False),
            egress=EgressProxy(resolve=web.resolve, block=web.block),
        )
        try:
            await pool.start()
        except Exception as err:
            pytest.skip(f"Chromium unavailable: {err}")
        try:
            recorder = HarBrowserService(pool, "record", tmp_path / "har")
            recorder.task_id = "kettle"
            await recorder.open("rec", allowed_domains=["shop.test"])
            assert (await recorder.execute("rec", Navigate(url=url))).ok
            await recorder.forget("rec")  # the HAR is written when the context closes
            assert recorder.path("kettle").exists()

            web.page("/price", "<h1>Blue Kettle</h1><p>Price: ฿9,999</p>")  # the site changed
            web.clear()
            replay = HarBrowserService(pool, "replay-strict", tmp_path / "har")
            replay.task_id = "kettle"
            await replay.open("rep", allowed_domains=["shop.test"])
            assert (await replay.execute("rep", Navigate(url=url))).ok
            text = (await replay.observe("rep")).render()
            assert "฿1,290" in text and "9,999" not in text
            assert web.hits == []  # served from the recording
            await replay.forget("rep")
        finally:
            await pool.stop()
