from __future__ import annotations

import json
import logging
import uuid
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .compare import compare_events, run_to_text, sse
from .config import Registry, Settings
from .gateway import Gateway
from .limits import LimitError, Limiter

STATIC = Path(__file__).parent / "static"
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("api")


class Params(BaseModel):
    temperature: float | None = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(800, ge=16)


class CompareReq(BaseModel):
    prompt: str
    system: str | None = None
    models: list[str]
    parameters: Params = Params()


class SummaryItem(BaseModel):
    model: str = Field(max_length=100)
    text: str


class SummaryReq(BaseModel):
    prompt: str
    results: list[SummaryItem]


class PreferenceReq(BaseModel):
    request_id: str = Field(max_length=64)
    model: str = Field(max_length=100)


def _err(status: int, code: str, message: str, retry_after: int | None = None) -> JSONResponse:
    headers = {"Retry-After": str(retry_after)} if retry_after else None
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


def _ip(request: Request) -> str:
    # Behind App Platform's proxy the client IP is the first X-Forwarded-For entry.
    # Spoofable if the app is exposed directly: acceptable for a PoC, noted in the README.
    xff = request.headers.get("x-forwarded-for", "")
    return xff.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def create_app(settings: Settings | None = None, registry: Registry | None = None,
               gateway: Gateway | None = None, limiter: Limiter | None = None) -> FastAPI:
    settings = settings or Settings()
    registry = registry or Registry()
    gateway = gateway or Gateway(settings)
    limiter = limiter or Limiter(settings)
    picks: Counter = Counter()
    app = FastAPI(title="Multi-model comparison PoC")
    app.state.limiter, app.state.picks, app.state.settings, app.state.registry = limiter, picks, settings, registry

    def active_models():
        return registry.for_mode(settings.mock_mode)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "mode": "mock" if settings.mock_mode else "live"}

    @app.get("/api/v1/models")
    async def models():
        reg = active_models()
        catalog_verified = None
        items = list(reg.values())
        if not settings.mock_mode:
            live = await gateway.live_catalog()
            catalog_verified = live is not None
            if live is not None:
                items = [m for m in items if m.id in live]
        defaults = [d for d in registry.default_ids if d in {m.id for m in items}] if not settings.mock_mode \
            else ["mock-fast", "mock-balanced", "mock-thorough"]
        return {
            "mode": "mock" if settings.mock_mode else "live",
            "catalog_verified": catalog_verified,
            "models": [m.public(registry.as_of) for m in items],
            "defaults": defaults,
            "pricing_as_of": registry.as_of, "pricing_source": registry.source,
            "limits": {"min_models": settings.min_models, "max_models": settings.max_models,
                       "max_prompt_chars": settings.max_prompt_chars,
                       "max_output_tokens": settings.max_output_tokens_cap,
                       "compares_per_hour": settings.compares_per_hour},
        }

    @app.post("/api/v1/compare")
    async def compare(req: CompareReq, request: Request, stream: bool = True):
        reg = active_models()
        prompt = req.prompt.strip()
        if not prompt:
            return _err(422, "empty_prompt", "Enter a prompt.")
        if len(prompt) > settings.max_prompt_chars or len(req.system or "") > settings.max_prompt_chars:
            return _err(422, "prompt_too_long", f"Prompt is limited to {settings.max_prompt_chars} characters.")
        ids = list(dict.fromkeys(req.models))
        if not settings.min_models <= len(ids) <= settings.max_models:
            return _err(422, "model_count", f"Choose {settings.min_models} to {settings.max_models} models.")
        unknown = [m for m in ids if m not in reg]
        if unknown:
            return _err(422, "unknown_model", f"Model not allowed: {unknown[0]}")
        cfgs = [reg[m] for m in ids]
        params = {"temperature": req.parameters.temperature,
                  "max_tokens": min(req.parameters.max_tokens, settings.max_output_tokens_cap)}
        ip = _ip(request)
        try:
            limiter.check_spend()
            limiter.check_rate(ip)
            hold = max((c.timeout_s or settings.default_timeout_s) for c in cfgs) + 10
            limiter.acquire(ip, hold)
        except LimitError as e:
            return _err(e.status, e.code, e.message, e.retry_after)

        request_id = "cmp_" + uuid.uuid4().hex[:10]
        events = compare_events(cfgs, gateway, settings, req.system or None, prompt, params, request_id)

        async def source():
            try:
                async for evt in events:
                    if evt["event"] == "done":
                        limiter.add_spend(evt.get("est_cost_usd"))
                    yield evt
            finally:
                await events.aclose()
                limiter.release(ip)

        if stream:
            async def body():
                async for evt in source():
                    yield sse(evt)
            return StreamingResponse(body(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        results: dict = {}
        async for evt in source():
            m = evt.get("model")
            if evt["event"] == "delta":
                results.setdefault(m, {"text": ""})["text"] += evt["text"]
            elif evt["event"] in ("done", "error"):
                results.setdefault(m, {"text": ""}).update({k: v for k, v in evt.items() if k not in ("event", "model")})
        return {"request_id": request_id, "results": results}

    @app.post("/api/v1/summarize")
    async def summarize(req: SummaryReq, request: Request):
        """Optional 'where do they agree and differ' view. One extra model call; not a verdict."""
        if not 2 <= len(req.results) <= settings.max_models:
            return _err(422, "model_count", "Need 2 to 4 responses to summarise.")
        ip = _ip(request)
        try:
            limiter.check_spend()
            limiter.check_rate(ip, "summary")
        except LimitError as e:
            return _err(e.status, e.code, e.message, e.retry_after)
        reg = active_models()
        sid = registry.summary_model if not settings.mock_mode else "mock-balanced"
        cfg = reg[sid]
        blocks = "\n\n".join(f"### Response {i + 1} ({r.model})\n{r.text[:4000]}" for i, r in enumerate(req.results))
        system = ("You compare several AI answers to the same prompt. Report only: (1) points all answers agree on, "
                  "(2) meaningful differences in content, approach or tone, (3) any claims that conflict or look doubtful. "
                  "Be concise, use short bullets, refer to responses by number. Do not name a winner.")
        out = await run_to_text(cfg, gateway, settings, system, f"Prompt:\n{req.prompt[:4000]}\n\n{blocks}",
                                {"temperature": 0.2, "max_tokens": 500}, "sum_" + uuid.uuid4().hex[:8])
        if out.get("status") != "success":
            return _err(502, "summary_failed", out.get("error") or "Summary failed.")
        limiter.add_spend(out.get("est_cost_usd"))
        return {"summary": out["text"], "model": sid, "est_cost_usd": out.get("est_cost_usd"), "generated_by_model": True}

    @app.post("/api/v1/preference", status_code=204)
    async def preference(req: PreferenceReq):
        picks[req.model] += 1
        log.info(json.dumps({"event": "preferred_pick", "request_id": req.request_id, "model": req.model}))

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()
