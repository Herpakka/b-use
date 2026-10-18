# Browser Use Agent

Natural-language task → an LLM (OpenAI, orchestrated with LangGraph) drives a real Chromium
(Playwright) until the task is done, pausing for a human on logins, captchas and irreversible
actions.

Status: **M3** — HTTP API with SSE, human-in-the-loop resume, live view, a debug page and a
Python SDK, on top of the browser layer and agent (M2). Phases 4–5 (hardening, deploy) are next.

Smoke eval on 2026-10-18 (`gpt-4.1` / `gpt-4.1-mini`, prompts `system@2`): 15/15 passed,
median 3 steps, mean $0.017 per task, p95 17 s. Runs killed mid-task resume with `bu resume`.

## Layers

```
api/      FastAPI            knows agent, store, bus     — never Playwright, prompts
agent/    LangGraph graph    knows action schema, BrowserService — never CDP, selectors, FastAPI
browser/  Playwright + DOM   knows Playwright            — never prompts, LLMs, HTTP
core/     config, logging, costs, store, bus             — knows none of the above
```

The rules are enforced by import-linter contracts in `pyproject.toml` (`uv run lint`).

## Run it (everything runs in Docker)

Playwright on Windows needs the Proactor event loop while async psycopg needs the Selector loop,
so the app, tests and CLI run in the Linux container.

```bash
cp .env.example .env                 # set BU_OPENAI_API_KEY and BU_API_KEYS
docker compose up -d --build         # app + postgres + redis
curl localhost:8000/healthz
```

Open <http://localhost:8000/debug>, paste a key from `BU_API_KEYS`, and start a run: you see each
step, the browser (screenshot or live view) and Approve/Deny/answer buttons when the agent waits
for you. API docs: <http://localhost:8000/docs>.

### See the browser on your desktop (optional)

By default the container runs headless Chromium. To watch the agent — and to log in or solve a
captcha yourself when it asks — run a visible Chromium on the host and let the container drive it:

```bash
uv run python -m devtools.browser_host --chrome "C:\path\to\chrome.exe"
```

Put the printed `BU_BROWSER_WS_ENDPOINT=…` line in `.env` and recreate the app
(`docker compose up -d --force-recreate app`). The server listens on 127.0.0.1 only and the
executable is fixed on the host side. Consider `BU_BLOCK_RESOURCES=false` so pages show images.

## API

All `/v1` routes need `Authorization: Bearer <key>` (or `X-API-Key`); a key maps to a tenant and
tenants never see each other's runs or browser sessions.

| Method | Path | |
|---|---|---|
| `POST` | `/v1/runs` | start a run → 202 (`task`, `allowed_domains`, `start_url?`, `session?`, `vision?`, `max_steps?`, `budget_usd?`) |
| `GET` | `/v1/runs`, `/v1/runs/{id}` | status, result, steps, cost, `pending` while `needs_human` |
| `GET` | `/v1/runs/{id}/stream` | SSE, one event per step; reconnect with `Last-Event-ID` |
| `GET` | `/v1/runs/{id}/events` | the same events as JSON |
| `POST` | `/v1/runs/{id}/resume` | `{"approved": bool, "answer": str}` while `needs_human` |
| `POST` | `/v1/runs/{id}/cancel` | stop the run and close its session |
| `GET` | `/v1/runs/{id}/screenshot` | live page, or the last one captured |
| `WS` | `/v1/runs/{id}/live`, `/v1/sessions/{name}/live` | JPEG frames (browsers: subprotocol `bu.key.<key>`) |
| `GET`/`DELETE` | `/v1/sessions`, `/v1/sessions/{name}` | open browser sessions |

Runs left running by a crashed or restarted server continue from their checkpoint on startup.

```python
from bu_client import Client

with Client("http://localhost:8000", api_key="…") as client:
    run = client.create_run("Price of 'A Light in the Attic'?", ["books.toscrape.com"])
    final = client.wait(run["id"], on_event=print, on_human=lambda pending: {"approved": True})
    print(final["result"])
```

## CLI, tests, evals

Tests (Chromium tests need the container):

```bash
docker compose run --rm app uv run pytest -m "not live and not llm"
```

Run a task from the CLI (checkpoints go to Postgres, so `bu resume` works after a restart):

```bash
docker compose run --rm app uv run bu run "What is the price of 'A Light in the Attic'?" --allow books.toscrape.com --start-url https://books.toscrape.com
```

```bash
docker compose run --rm app uv run bu resume <run_id>
```

Smoke eval (15 tasks on public practice sites):

```bash
docker compose run --rm app uv run python -m evals.runner evals/smoke.yaml
```

Lint/type-check also works on the Windows host: `uv sync` then `uv run lint`.

## Guardrails already in place

- Per-run domain allow-list enforced in the browser's request router (not in the prompt);
  `localhost`, private/link-local IPs and cloud metadata endpoints are always blocked.
- No `eval`/JavaScript action is exposed to the model.
- Page content reaches the model inside `<page_content>` delimiters and is declared untrusted.
- Clicks/submits that look irreversible (pay, buy, delete, send, ยืนยัน, ลบ, …) always pause for
  human approval — a code policy, not a prompt request.
- Three loop guards: `BU_MAX_STEPS`, `BU_BUDGET_USD`, `BU_RUN_TIMEOUT_S`; per-tenant rate limit,
  active-run limit and a cap on client-requested budgets.
- Text typed into password fields is masked in events. Credentials written into the task text
  itself are stored with the task — keep them out until secret placeholders land (phase 4.4).
