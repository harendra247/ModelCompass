"""Model gateway: one interface, two adapters (DigitalOcean inference, mock).

Adapters yield plain dicts so the compare service never sees HTTP details:
    {"type": "thinking"}                       reasoning tokens arrived (liveness only)
    {"type": "delta", "text": "..."}           visible answer text
    {"type": "usage", "input_tokens": n, "output_tokens": n}
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import time
from typing import AsyncIterator

import httpx

from .config import ModelCfg, Settings


class UpstreamError(Exception):
    """kind is one of: error, rate_limited, auth."""

    def __init__(self, kind: str, message: str, http_status: int | None = None, retry_after: float | None = None,
                 detail: str = ""):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.http_status = http_status
        self.retry_after = retry_after
        self.detail = detail.lower()  # raw upstream error text, used only to adapt the request


def _classify_http(status: int, body: str, retry_after: str | None) -> UpstreamError:
    snippet = body.strip().replace("\n", " ")[:200]
    if status == 429:
        try:
            ra = float(retry_after) if retry_after else None
        except ValueError:
            ra = None
        return UpstreamError("rate_limited", "Rate limited by the inference API (429)", status, ra)
    if status in (401, 403):
        return UpstreamError("auth", f"Not authorised for this model ({status})", status)
    if status == 404:
        return UpstreamError("error", f"Model not found or not enabled for this account (404) {snippet}", status)
    if status >= 500:
        return UpstreamError("error", f"Upstream server error ({status}) {snippet}", status)
    return UpstreamError("error", f"Request rejected ({status}) {snippet}", status, detail=body)


class DOAdapter:
    """OpenAI-compatible chat completions on DigitalOcean serverless inference."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client or httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
        )

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.settings.access_key}", "Content-Type": "application/json"}

    async def list_model_ids(self) -> set[str]:
        r = await self.client.get("/v1/models", headers=self._headers(), timeout=8.0)
        r.raise_for_status()
        return {m["id"] for m in r.json().get("data", [])}

    async def stream(self, cfg: ModelCfg, system: str | None, prompt: str, params: dict) -> AsyncIterator[dict]:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload: dict = {
            "model": cfg.id,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            cfg.max_tokens_param: params["max_tokens"],
        }
        if params.get("temperature") is not None:
            payload["temperature"] = params["temperature"]

        started = False
        retried = False
        adjustments: list[str] = []
        while True:
            try:
                async with self.client.stream("POST", "/v1/chat/completions", json=payload, headers=self._headers()) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        raise _classify_http(resp.status_code, body, resp.headers.get("retry-after"))
                    if adjustments:
                        yield {"type": "adjusted", "changes": list(adjustments)}
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if "error" in obj:
                            raise UpstreamError("error", f"Upstream error: {str(obj['error'])[:200]}")
                        for choice in obj.get("choices") or []:
                            if choice.get("finish_reason"):
                                yield {"type": "finish", "reason": choice["finish_reason"]}
                            delta = choice.get("delta") or {}
                            if delta.get("content"):
                                started = True
                                yield {"type": "delta", "text": delta["content"]}
                            elif delta.get("reasoning_content") or delta.get("reasoning"):
                                started = True
                                yield {"type": "thinking"}
                        usage = obj.get("usage")
                        if usage:
                            yield {"type": "usage", "input_tokens": usage.get("prompt_tokens"),
                                   "output_tokens": usage.get("completion_tokens")}
                    return
            except UpstreamError as exc:
                # Models differ in which parameters they accept. On a 400 that names one, drop or rename it and
                # try again (at most 3 times, before any output), and tell the UI what was changed.
                if exc.http_status == 400 and not started and len(adjustments) < 3:
                    change = _adapt_payload(payload, exc.detail)
                    if change:
                        adjustments.append(change)
                        continue
                retryable = exc.kind == "rate_limited" or (exc.http_status or 0) >= 500
                if not retried and retryable and not started:
                    retried = True  # one retry, only before any output; honour Retry-After but never wait long
                    await asyncio.sleep(min(exc.retry_after or 0.5, 3.0) + random.uniform(0, 0.25))
                    continue
                raise
            except httpx.ConnectError as exc:
                raise UpstreamError("error", f"Could not reach the inference API: {exc}") from exc
            except httpx.HTTPError as exc:
                raise UpstreamError("error", f"Network error: {type(exc).__name__}") from exc


def _adapt_payload(payload: dict, detail: str) -> str | None:
    """Fix the request in place based on a 400 message. Returns a short description, or None if nothing applies."""
    if "stream_options" in payload and "stream_options" in detail:
        del payload["stream_options"]
        return "no usage block requested"
    if "temperature" in payload and "temperature" in detail:
        del payload["temperature"]
        return "temperature not sent"
    if "max_tokens" in payload and "max_tokens" in detail:
        payload["max_completion_tokens"] = payload.pop("max_tokens")
        return "max_completion_tokens used"
    if "max_completion_tokens" in payload and "max_completion_tokens" in detail:
        payload["max_tokens"] = payload.pop("max_completion_tokens")
        return "max_tokens used"
    return None


_MOCK_PARAGRAPHS = [
    "Think of it as a short, practical explanation tuned to the prompt: start with the idea, then the trade-off, then one concrete example.",
    "The main point is that the right choice depends on what you optimise for. Faster models answer sooner; larger models usually reason more carefully; cheaper models make high volume affordable.",
    "A common mistake is to judge from a single answer. Run the same prompt a few times and look at how much the output varies before you decide.",
]


class MockAdapter:
    """Deterministic fake models for tests, offline demos and screenshots."""

    def __init__(self, speed: float = 1.0):
        self.speed = max(speed, 0.001)  # >1 makes the fakes faster (used by tests)

    async def stream(self, cfg: ModelCfg, system: str | None, prompt: str, params: dict) -> AsyncIterator[dict]:
        m = cfg.mock
        await asyncio.sleep(m.get("first_token_ms", 100) / 1000 / self.speed)
        if m.get("hang"):
            await asyncio.sleep(3600)
        if m.get("fail"):
            raise UpstreamError("error", "Simulated upstream error (500)", 500)
        if m.get("rate_limited"):
            raise UpstreamError("rate_limited", "Simulated rate limit (429)", 429)
        if m.get("on_start"):
            m["on_start"]()
        try:
            verbose = int(m.get("verbose", 1))
            body = f"**{cfg.label}** on: \"{prompt[:70]}\"\n\n" + "\n\n".join(_MOCK_PARAGRAPHS[: min(3, 1 + verbose)])
            body += "\n\n```python\nfor model in models:\n    print(model, run(prompt, model))\n```\n"
            words = body.split(" ")
            tps = max(1.0, float(m.get("tokens_per_s", 60)))
            step = 3
            stall_after = m.get("stall_after")
            for n, i in enumerate(range(0, len(words), step)):
                if stall_after is not None and n >= stall_after:
                    await asyncio.sleep(3600)  # goes silent mid-answer
                await asyncio.sleep(step / tps / self.speed)
                yield {"type": "delta", "text": " ".join(words[i:i + step]) + " "}
            yield {"type": "finish", "reason": "stop"}
            # tokenizers differ, so give each mock model its own chars-per-token ratio
            ratio = 3.2 + (sum(ord(c) for c in cfg.id) % 10) / 10
            yield {"type": "usage",
                   "input_tokens": math.ceil((len(prompt) + len(system or "")) / ratio),
                   "output_tokens": math.ceil(len(body) / ratio)}
        finally:
            if m.get("on_finish"):
                m["on_finish"]()


class Gateway:
    def __init__(self, settings: Settings, do: DOAdapter | None = None, mock: MockAdapter | None = None):
        self.settings = settings
        self.mock = mock or MockAdapter(settings.mock_speed)
        self.do = do if do is not None else (DOAdapter(settings) if settings.access_key else None)
        self._catalog: tuple[float, set[str] | None] | None = None

    def stream(self, cfg: ModelCfg, system, prompt, params):
        if cfg.api == "mock":
            return self.mock.stream(cfg, system, prompt, params)
        if cfg.api == "chat":
            if self.do is None:
                raise UpstreamError("auth", "No model access key configured")
            return self.do.stream(cfg, system, prompt, params)
        raise UpstreamError("error", f"API type {cfg.api!r} is not supported yet")

    async def live_catalog(self) -> set[str] | None:
        """IDs the account can really call, cached 5 min. None if it could not be fetched."""
        if self.do is None:
            return None
        if self._catalog and time.monotonic() - self._catalog[0] < 300:
            return self._catalog[1]
        try:
            ids: set[str] | None = await self.do.list_model_ids()
        except Exception:
            ids = None
        self._catalog = (time.monotonic(), ids)
        return ids
