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
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

from playwright._impl._driver import compute_driver_executable, get_driver_env

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
    }
    if chrome:
        options["executablePath"] = chrome
    return [node, "-e", _SERVER_JS, package, json.dumps(options)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="devtools.browser_host", description=__doc__)
    parser.add_argument("--chrome", help="Chromium executable (default: Playwright's bundled one)")
    parser.add_argument("--port", type=int, default=3010)
    parser.add_argument("--host", default="127.0.0.1", help="listen address (keep it loopback)")
    parser.add_argument("--headless", action="store_true", help="no visible windows")
    args = parser.parse_args(argv)

    if args.chrome and not Path(args.chrome).is_file():
        print(f"not a file: {args.chrome}", file=sys.stderr)
        return 2
    token = load_token()
    cmd = server_command(
        chrome=args.chrome, host=args.host, port=args.port, token=token, headless=args.headless
    )
    env = {**os.environ, **get_driver_env()}
    with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            if line.startswith("ready "):
                print("browser server ready")
                print("put this in .env, then recreate the app container:")
                print(f"  BU_BROWSER_WS_ENDPOINT=ws://host.docker.internal:{args.port}/{token}")
                print("Ctrl+C to stop", flush=True)
            else:
                print(line, end="", flush=True)
        try:
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            return 0


if __name__ == "__main__":
    sys.exit(main())
