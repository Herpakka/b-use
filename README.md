# Browser Use Agent

Natural-language task → an LLM (OpenAI, orchestrated with LangGraph) drives a real Chromium
(Playwright) until the task is done, pausing for a human on logins, captchas and irreversible
actions.

Status: **phase 4 (hardening) done** — secret placeholders, a context-aware approval policy,
monthly spend caps, Chromium crash recovery, PII redaction and retention, and an adversarial
suite, on top of the API (M3), the agent and the browser layer (M2). Phase 5 (eval at scale,
observability, deploy) is next.

Smoke eval on 2026-10-18 (`gpt-4.1` / `gpt-4.1-mini`, prompts `system@3`): 15/15 passed,
median 3 steps, mean $0.018 per task, p95 22 s. Adversarial suite: 15/15 scripted attacks held,
5/5 against the real model. Runs killed mid-task resume with `bu resume`.

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
cp .env.example .env                 # set BU_OPENAI_API_KEY, BU_API_KEYS (and BU_SECRETS_KEY)
docker compose up -d --build         # app + postgres + redis
curl localhost:8000/readyz           # 200 once database, bus, browser and OpenAI key are OK
```

`/healthz` is liveness (the process answers); `/readyz` is readiness (it can serve runs) and
says which dependency is down.

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
executable is fixed on the host side. It also runs the egress proxy on 127.0.0.1:3129, because
the browser's traffic now leaves from your machine: that is where your LAN, localhost and
metadata addresses must be refused. Keep it running while runs use the visible browser.
Consider `BU_BLOCK_RESOURCES=false` so pages show images.

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
| `PUT`/`GET`/`DELETE` | `/v1/secrets/{name}`, `/v1/secrets` | credentials for `{{SECRET:name}}` (values are never returned) |
| `GET` | `/v1/usage` | this month's spend, and what is left under the caps |

Runs left running by a crashed or restarted server continue from their checkpoint on startup.

```python
from bu_client import Client

with Client("http://localhost:8000", api_key="…") as client:
    run = client.create_run("Price of 'A Light in the Attic'?", ["books.toscrape.com"])
    final = client.wait(run["id"], on_event=print, on_human=lambda pending: {"approved": True})
    print(final["result"])
```

## Credentials: `{{SECRET:name}}`

Never put a password in the task. Store it, then refer to it by name:

```bash
curl -X PUT localhost:8000/v1/secrets/shop_pw -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" -d '{"value": "…", "domains": ["login.shop.example"]}'
```

A task like `Sign in with {{SECRET:shop_pw}} and download my last invoice` then works like a
password manager: the model only ever handles the placeholder; the browser layer types the
value, refuses any field outside `domains` (a phishing page on another host gets nothing), and
puts the placeholder back wherever the value shows up again — input values, page text, URLs,
Extract text. Values are encrypted at rest with `BU_SECRETS_KEY` (AES-GCM, bound to tenant and
name). The CLI and evals read `BU_LOCAL_SECRETS` instead. Not covered: a value the page
transforms before showing it (upper-cased, reformatted), and screenshots of non-password fields.

## Costs

Every run has a USD budget (`BU_BUDGET_USD`, or `budget_usd` up to `BU_MAX_BUDGET_USD`) and
stops when it is used up; `cost_breakdown` on the run shows where it went (plan, decide,
reflect, extract). Monthly caps (`BU_MONTHLY_BUDGET_USD` for the deployment,
`BU_TENANT_MONTHLY_BUDGET_USD` per tenant, UTC calendar month) are enforced when a run starts:
active runs count at their full budget, so the cap holds as long as runs stop at their budget
— which they do, give or take the last LLM call. A run that doesn't fit is cut down to what is
left, or refused with 402. The caps are checked in one process; several replicas would need a
shared counter (phase 5).

## Browser health

A probe asks Chromium for its version every 30 s. A crashed browser is relaunched at once; one
that stops answering twice in a row (frozen, not dead) is killed and relaunched. A crashed tab
is reopened on the same URL. Runs carry on: their next step re-opens the session from the
cookies saved at the last probe and returns to its last URL.

## Privacy and retention

Logs and the event log behind SSE mask e-mail addresses, phone, card and Thai national-ID
numbers and API keys (`BU_REDACT_PII`); the run record keeps the task, result and the pending
question as they are, because the client asked for them. Kept for: screenshots 24 h
(`BU_SCREENSHOT_TTL_S`), agent checkpoints (page snapshots) 24 h after the run ends
(`BU_CHECKPOINT_TTL_S`), whole runs with their events `BU_RUN_RETENTION_DAYS` (off by default).
A run's own browser session is deleted with its cookies when it ends; named sessions keep
theirs. Page content still goes to OpenAI — that is how the agent works.

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

Adversarial suite (phase 4 exit criteria) — hostile pages on a local fake internet against a
model that obeys every instruction on the page; the guards must hold anyway. `-m llm` runs the
same pages against the real model (~5 cents):

```bash
docker compose run --rm app uv run pytest tests/adversarial -m "not llm"
```

Smoke eval (15 tasks on public practice sites):

```bash
docker compose run --rm app uv run python -m evals.runner evals/smoke.yaml
```

Lint/type-check also works on the Windows host: `uv sync` then `uv run lint`.

## Guardrails

Enforced in code, not asked for in the prompt. `tests/adversarial/` attacks each one.

- **Domain allow-list (per run).** The request router refuses navigations — main frame,
  iframes, popups, form submissions — outside `allowed_domains`, and URLs that carry a link to
  another host in their query or fragment (the shape of an open redirect). Redirects are
  caught after the fact: after every action and before every observation the tab's URL is
  checked, and a tab that ended up outside the list goes back to its last allowed page (a
  popup is closed; an iframe is left out of the observation and of Extract). Limit: a redirect
  hidden server-side (`/go/123` → elsewhere) still sends that one request; nothing comes back.
  Subresources (CDNs, images, scripts) may come from any public host.
- **SSRF: egress proxy.** Every browser request — navigations, redirect hops, subresources,
  fetch/XHR, WebSockets — goes through `browser/egress.py`, which resolves the target itself,
  refuses loopback/private/link-local/reserved/metadata addresses, and connects to exactly the
  address it checked (no DNS rebinding window). The router also refuses IP literals and
  `localhost` names up front. Service workers are disabled so nothing bypasses either layer.
- **Prompt injection.** Page content (and dialogs, and Extract text) reaches the model between
  delimiters with a random boundary the page cannot know; tags inside are defused, and the task
  is restated after the page. No `eval`/JavaScript action exists. With a real model this is
  mitigation, not proof — which is why everything else on this list does not depend on it.
- **Human gate.** Clicks/Enter that pay, order, send, delete or confirm (Thai included); any
  form submit except search and login; any click in a payment form and "Continue"-style buttons
  on checkout pages; typing into payment fields (unless it is a stored secret) — all pause for
  approval, whatever the model wants.
- **Loop guards.** `BU_MAX_STEPS`, `BU_BUDGET_USD`, `BU_RUN_TIMEOUT_S`; repeated actions trigger
  reflection; monthly caps; per-tenant rate limit and active-run limit; observations are capped
  (`BU_OBSERVATION_MAX_CHARS`) so a huge page cannot flood the prompt.
- **Credentials** only as `{{SECRET:name}}` (above). Text typed into password fields is masked in
  events.
