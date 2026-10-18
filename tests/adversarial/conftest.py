from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest

from browser.egress import EgressProxy
from browser.pool import BrowserPool
from tests.adversarial.web import Web, serve_web
from tests.conftest import make_settings


@pytest.fixture(scope="module")
def web() -> Iterator[Web]:
    with serve_web() as w:
        yield w


@pytest.fixture(scope="module")
async def pool(web: Web, tmp_path_factory: pytest.TempPathFactory) -> AsyncIterator[BrowserPool]:
    egress = EgressProxy(resolve=web.resolve, block=web.block)
    p = BrowserPool(
        make_settings(tmp_path_factory.mktemp("adv"), block_resources=False), egress=egress
    )
    try:
        await p.start()
    except Exception as err:
        pytest.skip(f"Chromium unavailable: {err}")
    yield p
    await p.stop()


@pytest.fixture(autouse=True)
def fresh(web: Web) -> None:
    web.clear()
