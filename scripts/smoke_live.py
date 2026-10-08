"""Check the real DigitalOcean endpoint with your key. Run this first, from the project folder:

    python scripts/smoke_live.py            # reads DO_MODEL_ACCESS_KEY from the environment or .env

Prints (never the key itself): whether the key is loaded, whether the model list loads, which allowlisted
model IDs exist on your account, and for up to 3 models: first-token time, finish time, tokens, whether the
stream reported usage, and any parameters the app had to adjust. Costs a few cents.
"""
import asyncio
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.compare import compare_events  # noqa: E402
from app.config import Registry, Settings  # noqa: E402
from app.gateway import Gateway  # noqa: E402


def explain(status: int | None) -> str:
    return {
        401: "the key was rejected: make sure it is a Serverless Inference *model access key* (not a personal API token), with no extra characters",
        403: "the key is valid but not allowed to list models",
        404: "wrong base URL or path; expected https://inference.do-ai.run/v1/models",
        429: "rate limited: wait a minute and retry",
    }.get(status or 0, "unexpected response")


async def main() -> int:
    s = Settings()
    print(f"key loaded: {bool(s.access_key)} ({len(s.access_key or '')} chars)   base URL: {s.base_url}")
    if not s.access_key:
        print("No DO_MODEL_ACCESS_KEY found in the environment or in .env next to the app folder.")
        return 1
    reg, gw = Registry(), Gateway(s)

    print("\n1) GET /v1/models")
    try:
        live = await gw.do.list_model_ids()
        print(f"   OK: {len(live)} models on this account")
    except httpx.HTTPStatusError as exc:
        print(f"   FAILED HTTP {exc.response.status_code}: {explain(exc.response.status_code)}")
        return 2
    except httpx.HTTPError as exc:
        print(f"   FAILED to connect ({type(exc).__name__}). Check your internet/VPN/proxy settings.")
        return 2
    allow = list(reg.live.values())
    missing = [c.id for c in allow if c.id not in live]
    print("   allowlisted IDs missing on this account:", missing or "none")
    usable = [c for c in allow if c.id in live]
    if len(usable) < 2:
        print("   Fewer than 2 allowlisted models match. Edit app/models.json using IDs from the list above, e.g.:")
        print("   ", sorted(live)[:25])
        usable = allow

    print("\n2) One streamed call per model (up to 3)")
    bad = 0
    for c in usable[:3]:
        async for e in compare_events([c], gw, s, None, "Reply with one short sentence about Docker.",
                                       {"temperature": 0.2, "max_tokens": 60}, "smoke"):
            if e["event"] == "done":
                note = " (usage NOT reported by the stream: tokens estimated)" if e["tokens_approximate"] else ""
                adj = f" adjusted: {e['adjusted']}" if e.get("adjusted") else ""
                print(f"   OK   {c.id}: first token {e['ttft_ms']} ms, finish {e['latency_ms']} ms, "
                      f"tokens {e['input_tokens']}/{e['output_tokens']}, est ${e['est_cost_usd']}{note}{adj}")
            elif e["event"] == "error":
                bad += 1
                print(f"   FAIL {c.id}: {e['status']}: {e['error']}")
    print("\nResult:", "all good" if not bad else f"{bad} model call(s) failed: see above")
    return 0 if not bad else 3


sys.exit(asyncio.run(main()))
