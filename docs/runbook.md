# Runbook — Browser Use Agent

For whoever is on call. Each section: what you see → how to confirm → what to do. Commands are
for the staging stack (`docker/compose.staging.yml`); in another deployment the same signals
exist under the same names.

```bash
# shorthand used below
alias stg='docker compose -f docker/compose.staging.yml'
alias psql-bu='docker compose -f docker/compose.staging.yml exec postgres psql -U bu -d bu'
```

## Where to look

| What | Where |
|---|---|
| Can this replica serve runs, and if not why | `curl localhost:8088/readyz` → `checks.database / bus / browser / llm / draining / replica` (503 when not ready) |
| Is the process alive | `/healthz` (liveness only — restart the container if this fails) |
| Dashboards | Grafana <http://localhost:3001> → "Browser agent" (success rate, p50/p95 duration, cost per run, pool utilisation, spend per hour, browser restarts) |
| Raw metrics | Prometheus <http://localhost:9095>, metrics `bu_*` (scraped from each replica's `/metrics`, which the proxy does not expose) |
| One run, step by step | `GET /v1/runs/{id}/events`, the debug page `/debug`, or Langfuse (session = run id) |
| Logs (JSON, PII-masked) | `stg logs app --since 15m`; filter by `"event":"…"` names given below |
| Which replica owns a run | `GET /v1/runs/{id}` → `owner` (a container hostname) |

Useful PromQL:

```
sum(increase(bu_runs_finished_total{status="done"}[1h])) / sum(increase(bu_runs_finished_total[1h]))   # success rate
histogram_quantile(0.95, sum by (le) (rate(bu_run_duration_seconds_bucket[15m])))                      # p95 duration
sum(increase(bu_run_cost_usd_sum[1h])) / sum(increase(bu_run_cost_usd_count[1h]))                       # cost per run
bu_browser_sessions / bu_browser_capacity                                                               # pool use per replica
sum by (reason) (increase(bu_runs_refused_total[1h]))                                                   # 402 / 429 / 503 refusals
```

---

## 1. Browser pool exhausted

**You see:** `POST /v1/runs` answers **503** `all 4 browsers of this replica are busy` with
`Retry-After: 5`; `bu_runs_refused_total{reason="503"}` rising; "Browser pool utilisation" at
100 %.

**Why:** every running run *and every run paused for a human* holds a browser context on its
replica (`BU_MAX_CONTEXTS`, 4 by default). Paused runs keep theirs for up to
`BU_HELD_IDLE_TIMEOUT_S` (2 h) so a login window is still there when the person answers.

**Confirm:**

```bash
psql-bu -c "select owner, status, count(*) from bu_runs where status in ('running','needs_human','queued') group by 1,2 order by 1"
```

**Do:**
1. Mostly `needs_human`? Answer or cancel them — see §4. That frees the contexts at once.
2. Mostly `running` and they finish normally? Real load: add a replica, or raise
   `BU_MAX_CONTEXTS` together with the container memory limit (budget ~250 MB per context
   plus ~400 MB for the replica; the staging limit is 2.5 GB for 4).
3. A run `running` for far longer than `BU_RUN_TIMEOUT_S` (300 s) plus human waiting time is
   stuck: cancel it (`POST /v1/runs/{id}/cancel`). If it comes back, see §2.
4. Clients should honour `Retry-After`; through the proxy the retry usually lands on the other
   replica.

## 2. Chromium frozen, crashed or leaving zombies

**You see:** `readyz` → `checks.browser.ok: false` with `detail` "no answer within 10s" or
"not connected"; `bu_browser_up 0`; logs `chromium_restart`, `chromium_disconnected`,
`session_lost`; `bu_browser_restarts` climbing. Runs show a `stale` event and continue.

**What happens by itself:** the probe asks Chromium for its version every 30 s. A dead browser
is relaunched immediately; one that misses two probes is SIGKILLed and relaunched. Sessions are
re-opened from the cookies saved at the last probe and runs return to their last URL. A crashed
*tab* is reopened in place (observation note "the tab crashed and was reopened").

**Confirm and find the cause if it keeps happening (more than a restart an hour):**

```bash
stg logs app --since 1h | grep -E '"event": ?"(chromium_restart|chromium_disconnected|session_lost)"'
docker inspect --format '{{.Name}} OOMKilled={{.State.OOMKilled}} restarts={{.RestartCount}}' $(stg ps -q app)
docker exec browser-use-staging-app-1 sh -c 'df -h /dev/shm; ps -eo stat,comm | grep -c "^Z"'
docker stats --no-stream $(stg ps -q app)
```

- `/dev/shm` tiny (64 MB): the container lost `shm_size: 2gb` — tabs crash silently. Fix the
  compose/k8s spec.
- Memory at the limit / `OOMKilled=true`: too many contexts for the limit (§1) or a page that
  eats memory; lower `BU_MAX_CONTEXTS` or raise the limit.
- Zombies (`Z` count growing): the image's entrypoint is `tini`; something started the app
  without it (custom `entrypoint:`). Restore `tini` (or `init: true`).
- A specific site always crashes the tab: block it for that tenant's tasks.

**Do:** a rolling restart of the replica clears everything:
`docker stop -t 90 browser-use-staging-app-1 && docker start browser-use-staging-app-1`
(it drains first — §5).

## 3. Cost spike

**You see:** "LLM spend per hour" or `bu_run_cost_usd` p95 jumping; a tenant reports 402
`monthly budget exhausted`; the OpenAI bill.

**Confirm — who and what:**

```bash
psql-bu -c "select tenant, count(*), round(sum(cost_usd)::numeric,3) usd, round(avg(steps),1) steps from bu_runs where created_at > now() - interval '24 hours' group by 1 order by 3 desc"
psql-bu -c "select id, tenant, status, steps, round(cost_usd::numeric,4) usd, cost_breakdown, failure_reason from bu_runs where created_at > now() - interval '24 hours' order by cost_usd desc limit 10"
curl -s -H "Authorization: Bearer $KEY" localhost:8088/v1/usage
```

Read `cost_breakdown`: mostly `decide` with many steps → loops or huge pages; `reflect` high →
the agent keeps getting stuck on a site; `extract` high → long pages read repeatedly. Open the
top runs in Langfuse (session = run id) to see the prompts. `failure_reason` "budget … exhausted"
means the per-run cap did its job — the spike is many runs, not one.

**Usual causes:** a site changed and the agent loops until its budget; a client looping on
`POST /v1/runs`; `vision: true` runs (screenshots every step); a price change of the model
(`BU_MODEL_PRICES`); a prompt change (compare `prompt_version` in the nightly report).

**Do (from gentle to hard):**
1. Lower `BU_BUDGET_USD` (default per run) and `BU_MAX_BUDGET_USD` (what clients may ask for).
2. Set or lower `BU_TENANT_MONTHLY_BUDGET_USD` / `BU_MONTHLY_BUDGET_USD`. Active runs count at
   their full budget, so the cap holds; new runs get 402.
3. Lower `BU_RATE_LIMIT_PER_MINUTE` / `BU_MAX_ACTIVE_RUNS_PER_TENANT`.
4. Stop a tenant: remove its key from `BU_API_KEYS` and restart the replicas one at a time.
5. Stop everything: `BU_MONTHLY_BUDGET_USD=0`, rolling restart. Running runs finish their
   current budget at most.

## 4. Stuck human prompt

**You see:** runs in `needs_human` for hours; the pool fills up (§1); users say "the agent is
waiting and nothing happens when I click Approve".

**Confirm:**

```bash
psql-bu -c "select id, tenant, owner, lease_until > now() as owner_alive, now() - updated_at as waiting, pending->>'kind' kind, left(pending->>'reason', 60) reason, command from bu_runs where status = 'needs_human' order by updated_at"
```

- `command` is set and stays set: a resume/cancel was forwarded to the owner, which hasn't
  picked it up. If `owner_alive` is false the owner is gone; any replica adopts the run within
  one lease (`BU_RUN_LEASE_S`, 30 s) and carries the command out — check
  `bu_runs_adopted_total` and the run's `recovered` event. If `owner_alive` stays true but
  nothing happens, that replica's command loop is stuck: logs `command_listener_failed` /
  `lease_upkeep_failed`; restart that replica (it drains and releases its leases).
- No `command`: nobody has answered. Answer through the debug page or the API
  (`POST /v1/runs/{id}/resume` with `{"approved": true|false, "answer": "…"}`), or cancel.
- The person needs the browser itself (captcha, 2FA): the live view only works on the owning
  replica; the debug page sticks to one replica by cookie, so open it until `/readyz` shows the
  run's `owner`, or answer with text instead.

After a replica dies, an adopted paused run's page is reopened at its last URL from saved
cookies; a half-filled form is lost — the agent re-fills it after the answer.

## 5. A replica is down, restarting or deploying

**What happens:** the proxy health-checks `/readyz` every 5 s and retries a request on the other
replica for up to 15 s. On SIGTERM a replica refuses new runs (503 "shutting down"), lets running
runs finish for up to `BU_DRAIN_TIMEOUT_S` (60 s), then stops the rest and releases its leases —
the other replica continues them from their checkpoints at once. SSE streams are cut after 10 s;
clients reconnect with `Last-Event-ID` and continue on the other replica. A replica that dies
without draining (SIGKILL, OOM, power) leaves its leases to expire: its runs resume elsewhere
within ~30 s.

**Rolling deploy (one at a time, never both):**

```bash
docker compose -f docker/compose.staging.yml build app
for c in browser-use-staging-app-1 browser-use-staging-app-2; do
  docker stop -t 90 $c && docker compose -f docker/compose.staging.yml up -d --no-deps --no-recreate app
  until curl -fs localhost:8088/readyz >/dev/null; do sleep 2; done
done
```

**Check after any of this:** `uv run python -m devtools.staging_check` (failover, drain and
forwarding with real runs, a few cents).

## 6. Database or Redis unavailable

- **Postgres** (`readyz` → `database` not ok): nothing works — runs can't checkpoint and fail.
  Restore Postgres; runs that were `running` continue from their last checkpoint once a replica
  can adopt them.
- **Redis** (`readyz` → `bus` not ok): runs keep going and events are still stored, but live SSE
  tails stop (clients reconnect and replay from the store), forwarded commands wait for the
  owner's next lease tick (~10 s) instead of arriving at once, and the rate limiter fails open
  (log `ratelimit_unavailable`).

## 7. Other things that look like outages

| Symptom | Meaning | Do |
|---|---|---|
| `readyz` → `llm: BU_OPENAI_API_KEY is not set` | runs are refused with 503 | set the key, restart |
| Step lines "navigation blocked: … not in the allowed domains" | working as intended | the client must add the domain to `allowed_domains` |
| "… refused by the egress proxy (internal address)" | the site resolves to a private address, or the remote browser's proxy (`devtools.browser_host`) is not running | check DNS / start browser_host |
| `TypeText → refused: no secret named 'x'` / 422 `unknown secret` | the task uses `{{SECRET:x}}` that isn't stored for that tenant | `PUT /v1/secrets/x` |
| 503 `secrets need BU_SECRETS_KEY` | the key is unset | set it; secrets stored under another key are unreadable (log `secret_undecryptable`) — re-store them |
| Langfuse shows nothing | tracing is off (no `LANGFUSE_*` keys) or Langfuse is down; runs are unaffected | check `tracing_enabled` / `tracing_unavailable` in the logs |
| 402 `monthly budget exhausted` | the cap is reached (§3) | raise the cap or wait for the next month (UTC) |
