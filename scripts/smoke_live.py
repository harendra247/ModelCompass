"""Run once with a real key before demoing: checks the assumptions the mock cannot.

    DO_MODEL_ACCESS_KEY=... python scripts/smoke_live.py

Prints which allowlisted model IDs exist on the account, whether streaming returns usage,
and TTFT / latency per model. Costs a few cents.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.compare import compare_events
from app.config import Registry, Settings
from app.gateway import Gateway


async def main():
    s = Settings()
    if not s.access_key:
        sys.exit("Set DO_MODEL_ACCESS_KEY")
    reg, gw = Registry(), Gateway(s)
    live = await gw.live_catalog()
    print("catalog fetched:", live is not None)
    cfgs = list(reg.live.values())
    if live:
        print("allowlisted but NOT on this account:", [c.id for c in cfgs if c.id not in live] or "none")
        cfgs = [c for c in cfgs if c.id in live]
    for c in cfgs[:3]:
        async for e in compare_events([c], gw, s, None, "Reply with one short sentence about Docker.",
                                       {"temperature": 0.2, "max_tokens": 60}, "smoke"):
            if e["event"] in ("done", "error"):
                print({k: v for k, v in e.items() if k != "event"})
    print("If tokens_approximate is True, the API did not return usage in the stream.")


asyncio.run(main())
