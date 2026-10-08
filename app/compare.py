"""Parallel fan-out with per-model failure isolation.

One asyncio task per model. Each task turns every outcome (success, timeout, upstream error)
into events on a shared queue, so one model can never fail the whole comparison.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Callable

from .config import ModelCfg, Settings
from .gateway import Gateway, UpstreamError

log = logging.getLogger("compare")


def est_cost(cfg: ModelCfg, input_tokens: int | None, output_tokens: int | None) -> float | None:
    if cfg.input_per_m is None or cfg.output_per_m is None or input_tokens is None or output_tokens is None:
        return None
    return round((input_tokens * cfg.input_per_m + output_tokens * cfg.output_per_m) / 1_000_000, 6)


class FirstTokenTimeout(Exception):
    pass


async def run_model(cfg: ModelCfg, gateway: Gateway, settings: Settings, system: str | None, prompt: str,
                    params: dict, emit: Callable, request_id: str) -> None:
    total_s = cfg.timeout_s or settings.default_timeout_s
    first_s = cfg.first_token_timeout_s or settings.default_first_token_timeout_s
    t0 = time.perf_counter()
    ms = lambda: int((time.perf_counter() - t0) * 1000)  # noqa: E731
    chars, ttft, usage, saw_thinking, got_any = 0, None, None, False, False
    status, error = "success", None
    agen = None
    await emit({"event": "status", "model": cfg.id, "status": "running"})
    try:
        async with asyncio.timeout(total_s):
            agen = gateway.stream(cfg, system, prompt, params)
            it = agen.__aiter__()
            while True:
                try:
                    if got_any:
                        chunk = await it.__anext__()
                    else:
                        try:
                            chunk = await asyncio.wait_for(it.__anext__(), first_s)
                        except asyncio.TimeoutError:
                            raise FirstTokenTimeout()
                except StopAsyncIteration:
                    break
                got_any = True
                if chunk["type"] == "delta":
                    if ttft is None:
                        ttft = ms()
                    chars += len(chunk["text"])
                    await emit({"event": "delta", "model": cfg.id, "text": chunk["text"]})
                elif chunk["type"] == "thinking" and not saw_thinking:
                    saw_thinking = True
                    await emit({"event": "status", "model": cfg.id, "status": "thinking"})
                elif chunk["type"] == "usage":
                    usage = chunk
    except FirstTokenTimeout:
        status, error = "timeout", f"No response started within {first_s:g}s"
    except TimeoutError:
        status, error = "timeout", f"Did not finish within {total_s:g}s"
    except UpstreamError as exc:
        status = "rate_limited" if exc.kind == "rate_limited" else "error"
        error = exc.message
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # never let one model take the request down
        status, error = "error", f"Unexpected {type(exc).__name__}"
        log.exception("unexpected error", extra={"model": cfg.id})
    finally:
        if agen is not None:
            try:
                await agen.aclose()
            except BaseException:
                pass

    latency = ms()
    if status == "success":
        approx = not (usage and usage.get("input_tokens") is not None and usage.get("output_tokens") is not None)
        in_tok = (usage or {}).get("input_tokens")
        out_tok = (usage or {}).get("output_tokens")
        if approx:  # upstream sent no usage: estimate ~4 chars/token and say so
            in_tok = (len(prompt) + len(system or "")) // 4
            out_tok = chars // 4
        evt = {"event": "done", "model": cfg.id, "status": "success", "latency_ms": latency, "ttft_ms": ttft,
               "input_tokens": in_tok, "output_tokens": out_tok, "tokens_approximate": approx,
               "chars": chars, "est_cost_usd": est_cost(cfg, in_tok, out_tok)}
    else:
        evt = {"event": "error", "model": cfg.id, "status": status, "error": error,
               "latency_ms": latency, "ttft_ms": ttft, "chars": chars}
    # metadata only: never log prompt or output text
    log.info(json.dumps({"request_id": request_id, **{k: v for k, v in evt.items() if k != "event"},
                         "event": "model_result"}))
    await emit(evt)


async def compare_events(cfgs: list[ModelCfg], gateway: Gateway, settings: Settings, system: str | None,
                         prompt: str, params: dict, request_id: str) -> AsyncIterator[dict]:
    q: asyncio.Queue = asyncio.Queue()

    async def emit(e: dict):
        await q.put(e)

    yield {"event": "start", "request_id": request_id, "models": [c.id for c in cfgs]}
    tasks = [asyncio.create_task(run_model(c, gateway, settings, system, prompt, params, emit, request_id)) for c in cfgs]

    async def closer():
        await asyncio.gather(*tasks, return_exceptions=True)
        await q.put(None)

    closer_task = asyncio.create_task(closer())
    try:
        while (evt := await q.get()) is not None:
            yield evt
        yield {"event": "complete", "request_id": request_id}
    finally:
        # client went away (or finished): cancel anything still running so upstream calls stop
        for t in tasks:
            t.cancel()
        closer_task.cancel()
        await asyncio.gather(*tasks, closer_task, return_exceptions=True)


async def run_to_text(cfg: ModelCfg, gateway: Gateway, settings: Settings, system: str | None, prompt: str,
                      params: dict, request_id: str) -> dict:
    """Run one model to completion and return its text plus the final metrics event."""
    parts: list[str] = []
    final: dict = {}

    async def emit(e: dict):
        if e["event"] == "delta":
            parts.append(e["text"])
        elif e["event"] in ("done", "error"):
            final.update(e)

    await run_model(cfg, gateway, settings, system, prompt, params, emit, request_id)
    return {"text": "".join(parts), **final}


def sse(evt: dict) -> str:
    name = evt["event"]
    data = {k: v for k, v in evt.items() if k != "event"}
    return f"event: {name}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
