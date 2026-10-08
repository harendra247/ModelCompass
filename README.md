# Model Compare: one prompt, several DigitalOcean models, one view

A proof of concept for "Model FOMO". A user enters one prompt, picks 2 to 4 models hosted on
DigitalOcean serverless inference, and sees the answers stream in side by side with
latency, time to first token, tokens and an estimated cost. The user marks the answer they prefer.

The full reasoning is in the PRD (`PRD_v4_...docx`). This README covers what is built, how to run it,
and the trade-offs.

## Test it without a model access key

You do not need a key. With no `DO_MODEL_ACCESS_KEY`, the app starts in **demo mode**: the UI shows a
"Demo mode: simulated models" badge and offers five fake models (fast, balanced, thorough, one that always errors,
one that hangs and times out). Everything except the real network call is the real code path: parallel fan-out,
streaming, per-model failure, timeouts, cost, limits, Stop, Run again, export, the optional summary.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q                        # 42 tests, about 7 s, no network, no key
.venv/bin/uvicorn app.main:app --port 8080           # open http://localhost:8080
```

Things to try in demo mode: pick Mock Fast + Mock Thorough + Mock Flaky (one card errors, the others finish);
pick Mock Hang (times out after 4 s with partial state); press Stop mid-run; press Run again and read the variance table;
tick the differences summary; export JSON. Set `MOCK_MODE=1` to force demo mode even if a key is present.

What demo mode cannot prove: that the real endpoint accepts these model IDs, streams usage, and behaves as the docs say.
When you get a key, run `scripts/smoke_live.py` first, then start the app with the key set.

```bash
# Put the key in .env (git-ignored): DO_MODEL_ACCESS_KEY=...   The app loads .env itself (a real env var wins).
.venv/bin/python scripts/smoke_live.py               # checks key, model IDs, usage in stream, TTFT (costs a few cents)
.venv/bin/uvicorn app.main:app --port 8080           # startup log says: "mode": "live", "key_loaded": true
```

If the badge says "Demo mode" with a key set, the key was not found: check the startup log line. If cards show
"Request rejected (400)", the model rejected a parameter the app could not adapt: the message names it.
The gateway already adapts once for `temperature`, `stream_options` and `max_tokens` vs `max_completion_tokens`,
and the card shows "Adjusted for this model: ...".

## Docker

One image serves both the API and the front end. The UI is static files in `app/static`, so a second front-end
container would only add a proxy hop and a second thing to deploy. If the UI later becomes a React build, split it:
a `frontend` service (build, then serve with nginx) in front of the same `app` service.

```bash
cp .env.example .env                                 # optional; leave the key empty for demo mode
docker compose up --build                            # http://localhost:8080
DO_MODEL_ACCESS_KEY=... docker compose up --build    # live mode (or put the key in .env)
docker compose --profile dev up dev                  # http://localhost:8081, code reload, demo mode by default
```

The image runs as a non-root user with a read-only filesystem, one worker (limits are in memory), and a health check on `/healthz`.
`.env` is git-ignored; `.env.example` lists every setting. The compose file was validated with `docker compose config`;
the image itself was not built in the authoring environment (no Docker daemon), so run `docker compose up --build` once and
open the page before relying on it.

Deploy: `.do/app.yaml` is an App Platform spec (set the repo, add the key as a secret).

**Status of verification.** Everything below is tested against the built-in mock models and against a fake
HTTP server that imitates the OpenAI-compatible stream. It has **not** been run against the real DigitalOcean
endpoint (no key in the build environment). The Dockerfile, compose file and App Platform spec have not been built or deployed here.

## What is built (the mandatory slice)

| Area | Built |
|---|---|
| Fan-out | One asyncio task per model; total time is the slowest model, not the sum (tested) |
| Failure isolation | Each model ends as `success`, `error`, `timeout` or `rate_limited`; the others are unaffected (tested) |
| Streaming | One multiplexed SSE stream; blocking JSON fallback with `?stream=false` |
| Timeouts | Stall-based: no first token in 20 s, or no output for 20 s mid-answer, ends the card with its partial text. A 120 s hard cap is only a backstop. All per-model config / env (`FIRST_TOKEN_TIMEOUT_S`, `IDLE_TIMEOUT_S`, `MODEL_TIMEOUT_S`) |
| Retries | One jittered retry on 429 and 5xx, only before any output, honouring `Retry-After` |
| Metrics | Latency, time to first token, input and output tokens (per model), estimated cost |
| Cost | Price table in `app/models.json`, with source URL and as-of date; shown as an estimate |
| Combined view | Headline strip (fastest first token, fastest finish, cheapest, longest) plus optional "agree and differ" summary |
| Variance | Run again keeps every run in a tab and shows finish times across runs |
| Human verdict | "Prefer this one" per run, logged as an event |
| Controls | Stop, export JSON and Markdown, copy |
| Safety | Key stays on the server; model allowlist; prompt and output-token caps; per-IP hourly limit and concurrency limit; global daily spend ceiling |
| Privacy | No prompts or answers stored; logs hold metadata only |
| Testability | Mock adapter with configurable delay, failure, hang; also the offline demo mode |

## Design decisions and trade-offs

| Decision | Chosen | Alternative | What we give up |
|---|---|---|---|
| Shape of the service | One stateless FastAPI process, no database, queue or cache | Gateway + queue + workers + Redis | No history, no replay after a dropped connection. Fine for a first look; the event contract is what we keep stable |
| Calls to models | Parallel async tasks | Sequential | Bursty load on the account and a higher chance of hitting rate limits. Mitigated by caps and one retry |
| Delivery | SSE over `fetch` (POST), with a blocking fallback | Blocking only, or WebSockets | Partial-state handling in the UI. `EventSource` cannot POST, so the client parses the stream itself |
| Frontend | One HTML file, vanilla JS, markdown libraries vendored locally | React + build step | Less structure as the UI grows (this is the first thing to migrate in Phase 2). Gained: one container, nothing to build, works offline |
| Model access | One adapter for the OpenAI-compatible chat API, per-model config for quirks | One adapter class per provider | Config needs upkeep as models change. Models that only support the Responses API are not selectable yet |
| Who judges | The human picks; an optional model-written summary shows differences | Automatic LLM-as-judge ranking | No automatic ranking. A judge adds cost, latency and its own bias; DigitalOcean Evaluations already does dataset-scale judging |
| Summary step | Optional, off by default, labelled as model-written, never names a winner | Always on | One extra model call and a model's own blind spots. Kept because "combined view" can reasonably mean more than side-by-side |
| Cost | Static price table with as-of date | Fetch from an API | Prices can drift. Shown as an estimate with the date; lowest prompt-length tier is used |
| Tokens | Shown per model, with `~` when estimated | One comparable total | Cannot rank models by tokens, because tokenizers differ. Cost is the comparable number |
| Quality signal | One sample per model per run | Several samples by default | Single answers are noisy. Run again shows the spread; multi-sample runs belong in the evaluation phase |
| Credentials | One shared server-side key with limits | Bring your own key | Spend is ours. Bounded by the per-IP limit, output cap and daily ceiling |
| Limits | In memory, per process, per IP | Redis, per authenticated user | Reset on restart, not shared across instances, IP can be spoofed or shared. That is why the spec pins one instance |
| Spend ceiling | Checked before each comparison | Reserve the worst-case cost | Concurrent in-flight requests can overshoot the ceiling slightly |
| Persistence | None | Store every comparison | No history or sharing until there is a privacy decision on prompts |
| Model count | 2 to 4, default 3, config value | Unlimited | Less exploration per run; readable UI and bounded cost |
| Packaging | One container serves the API and the static UI | Separate front-end and back-end containers behind a proxy | Cannot scale or cache the UI separately. Split when the UI becomes a React build |
| Model list | Allowlist in config, intersected with `GET /v1/models` | Show whatever the API returns | The allowlist needs upkeep; the benefit is no embeddings, image or audio models in a chat picker |

## Scope: now, next, later

**Built now** (above).

**Next (days, not hours).** Responses-API adapter for models that only support it; image/vision input;
auth and a per-user quota; Redis-backed limits; React migration; saved comparisons with expiry;
shareable links; token usage verified against the live API; automated pricing sync if the platform exposes it;
a task-first picker ("I need SQL generation") built from preferred picks.

**Later (needs real traction).** Prompt-set evaluations on top of DigitalOcean Evaluations, with multi-sample
runs and an LLM judge; recommendations with a quality-latency-cost frontier; "create an Inference Router from
these results"; change alerts when a new model or price beats the user's current choice;
shadow-testing production traffic; a regression set for safe model switching. Scaling stages for millions
of users are in the PRD.

## Known limitations

- Not verified against the live endpoint (see above). IDs and prices were read from DigitalOcean docs on 2026-10-08.
- Streaming usage: if the API sends no usage block, tokens are estimated at ~4 characters per token and marked `~`.
- The "thinking" status relies on the model sending `reasoning_content` or `reasoning` deltas. Time to first token measures the first visible answer text, so reasoning models look slower by design.
- Prompts of 8,000 characters or fewer, output capped at 4,000 tokens (default request 1,500), text only.
- Timeouts are about silence, not total time: a slow model that keeps streaming is allowed to finish (up to the 120 s hard cap). A call that stalls keeps its partial text, shows an estimated token count and cost (marked `~`), and counts toward the spend ceiling. Answers cut off by the max-output-tokens setting are labelled on the card.

## Layout

```
app/main.py      HTTP API, validation, limits wiring
app/compare.py   fan-out, per-model isolation, SSE encoding
app/gateway.py   DigitalOcean adapter + mock adapter
app/limits.py    rate, concurrency and spend controls
app/config.py    settings and model registry
app/models.json  allowlist, prices, as-of date
app/static/      single-page UI (served by the same container)
Dockerfile, docker-compose.yml, .env.example, .gitignore, .dockerignore, .do/app.yaml
tests/           42 tests: parallelism, isolation, timeouts, cancellation, adapter parsing, limits
```
