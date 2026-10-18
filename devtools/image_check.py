"""Check a built image: the app imports, and its Chromium launches and renders a page.

docker run --rm --shm-size=2g --entrypoint python browser-use:prod -m devtools.image_check
"""

from __future__ import annotations

import asyncio
import sys

from playwright.async_api import async_playwright


async def main() -> int:
    import api.main  # noqa: F401  (the whole app imports with the installed dependencies)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=["--disable-dev-shm-usage"])
        page = await browser.new_page()
        await page.set_content("<h1 id=t>ok</h1>")
        text = await page.text_content("#t")
        print(f"chromium {browser.version}: {text}")
        await browser.close()
    return 0 if text == "ok" else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
