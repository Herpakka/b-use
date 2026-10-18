# Browser Use Agent

Natural-language task → an LLM (OpenAI, orchestrated with LangGraph) drives a real Chromium
(Playwright) until the task is done, pausing for a human on logins, captchas and irreversible
actions.

Status: **M2** — browser layer + agent, driven from the `bu` CLI. The HTTP API/SSE (phase 3)
is not built yet; `GET /healthz` is the only route.

Smoke eval on 2026-10-18 (`gpt-4.1` / `gpt-4.1-mini`, prompts `system@2`): 15/15 passed,
median 3 steps, mean $0.017 per task, p95 17 s. Runs killed mid-task resume with `bu resume`.

## Layers

```
api/      FastAPI            knows agent, store, bus     — never Playwright, prompts
agent/    LangGraph graph    knows action schema, BrowserService — never CDP, selectors, FastAPI
browser/  Playwright + DOM   knows Playwright            — never prompts, LLMs, HTTP
core/     config, logging, costs                         — knows none of the above
```

The rules are enforced by import-linter contracts in `pyproject.toml` (`uv run lint`).

## Run it (everything runs in Docker)

Playwright on Windows needs the Proactor event loop while async psycopg needs the Selector loop,
so the app, tests and CLI run in the Linux container.

```bash
cp .env.example .env                 # then put your key in BU_OPENAI_API_KEY
docker compose up -d --build         # app + postgres + redis
curl localhost:8000/healthz
```

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
- Three loop guards: `BU_MAX_STEPS`, `BU_BUDGET_USD`, `BU_RUN_TIMEOUT_S`.
