"""Run a visible Chromium on this machine for the containerised app to drive.

    uv run python -m devtools.browser_host --chrome C:\\path\\to\\chrome.exe

Starts Playwright's browser server (the same Playwright version as the app, from the bundled
driver) around a local Chromium and prints the ``BU_BROWSER_WS_ENDPOINT`` line for ``.env``.

Safety:
* The executable is chosen here, on the host. A client can only open contexts in the browser
  this script launched; it cannot start other programs.
* The server listens on 127.0.0.1 by default (Docker Desktop forwards ``host.docker.internal``
  to the host's loopback), so nothing else on the network can reach it.
* The endpoint path is a random token kept in ``.data/browser-host-token``, so the URL stays
  stable across restarts and other local processes cannot guess it.
* Alongside the browser it runs the egress proxy (browser/egress.py) on 127.0.0.1:3129: the
  browser's traffic leaves from *this* machine, so this is where internal addresses — your
  LAN, localhost, cloud metadata — must be refused. Contexts authenticate with the token.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import secrets
import sys
from pathlib import Path

from playwright._impl._driver import compute_driver_executable, get_driver_env

from browser.egress import EgressProxy

TOKEN_FILE = Path(".data/browser-host-token")

_SERVER_JS = r"""
const { chromium } = require(process.argv[1]);
const options = JSON.parse(process.argv[2]);
(async () => {
  const server = await chromium.launchServer(options);
  console.log("ready " + server.wsEndpoint());
  const stop = async () => { await server.close(); process.exit(0); };
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);
})().catch((err) => { console.error(err); process.exit(1); });
"""


def load_token(path: Path = TOKEN_FILE) -> str:
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_urlsafe(24)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token, encoding="utf-8")
    return token


def server_command(
    *, chrome: str | None, host: str, port: int, token: str, headless: bool
) -> list[str]:
    node, cli = compute_driver_executable()
    package = str(Path(cli).parent)
    options: dict[str, object] = {
        "headless": headless,
        "host": host,
        "port": port,
        "wsPath": f"/{token}",
        # Chromium on Windows only honours per-context proxies (the app's egress proxy) when
        # launched with a global one; every context overrides this placeholder.
        "proxy": {"server": "http://per-context"},
    }
    if chrome:
        options["executablePath"] = chrome
    return [node, "-e", _SERVER_JS, package, json.dumps(options)]


async def serve(
    *, chrome: str | None, host: str, port: int, egress_port: int, headless: bool
) -> int:
    token = load_token()
    egress = EgressProxy(host="127.0.0.1", port=egress_port, password=token)
    await egress.start()
    cmd = server_command(chrome=chrome, host=host, port=port, token=token, headless=headless)
    proc = await asyncio.create_subprocess_exec(
        *cmd, env={**os.environ, **get_driver_env()}, stdout=asyncio.subprocess.PIPE
    )
    assert proc.stdout is not None
    try:
        async for raw in proc.stdout:
            line = raw.decode(errors="replace")
            if line.startswith("ready "):
                print(f"browser server ready; egress proxy on 127.0.0.1:{egress.port}")
                print("put this in .env, then recreate the app container:")
                print(f"  BU_BROWSER_WS_ENDPOINT=ws://host.docker.internal:{port}/{token}")
                if egress.port != 3129:
                    print(f"  BU_EGRESS_BROWSER_URL=http://127.0.0.1:{egress.port}")
                print("Ctrl+C to stop", flush=True)
            else:
                print(line, end="", flush=True)
        return await proc.wait()
    finally:
        if proc.returncode is None:
            proc.terminate()
            with contextlib.suppress(Exception):
                await proc.wait()
        await egress.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="devtools.browser_host", description=__doc__)
    parser.add_argument("--chrome", help="Chromium executable (default: Playwright's bundled one)")
    parser.add_argument("--port", type=int, default=3010)
    parser.add_argument("--host", default="127.0.0.1", help="listen address (keep it loopback)")
    parser.add_argument("--egress-port", type=int, default=3129)
    parser.add_argument("--headless", action="store_true", help="no visible windows")
    args = parser.parse_args(argv)

    if args.chrome and not Path(args.chrome).is_file():
        print(f"not a file: {args.chrome}", file=sys.stderr)
        return 2
    try:
        return asyncio.run(
            serve(
                chrome=args.chrome,
                host=args.host,
                port=args.port,
                egress_port=args.egress_port,
                headless=args.headless,
            )
        )
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
