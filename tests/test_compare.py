import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app.compare import compare_events, est_cost
from app.config import ModelCfg, Registry, Settings
from app.gateway import DOAdapter, Gateway
from app.limits import Limiter
from app.main import create_app


def mk(id, price=(1.0, 2.0), **kw):
    mock = kw.pop("mock", {})
    return ModelCfg(id=id, label=id, provider="t", context=1000, api="mock",
                    input_per_m=price[0], output_per_m=price[1], mock=mock, **kw)


def settings(**kw):
    s = Settings(access_key=None)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


async def collect(cfgs, s=None, **kw):
    s = s or settings()
    gw = Gateway(s)
    out = []
    async for e in compare_events(cfgs, gw, s, None, "hello world", {"temperature": 0.7, "max_tokens": 100}, "cmp_t"):
        out.append(e)
    return out


def by_model(events, name):
    return [e for e in events if e.get("event") == name]


# ---------- core: parallelism, isolation, timeouts, cancellation ----------

def test_runs_in_parallel_not_sequentially():
    cfgs = [mk(f"m{i}", mock={"first_token_ms": 300, "tokens_per_s": 10000}) for i in range(3)]
    t0 = time.perf_counter()
    events = asyncio.run(collect(cfgs))
    wall = time.perf_counter() - t0
    assert len(by_model(events, "done")) == 3
    assert wall < 0.6, f"sequential would be ~0.9s, got {wall:.2f}s"  # slowest model, not the sum


def test_orchestration_overhead_is_small():
    cfgs = [mk(f"m{i}", mock={"first_token_ms": 200, "tokens_per_s": 100000}) for i in range(4)]
    t0 = time.perf_counter()
    asyncio.run(collect(cfgs))
    assert time.perf_counter() - t0 - 0.2 < 0.3  # AC2: under 300 ms over the slowest model


def test_one_failure_does_not_affect_others():
    cfgs = [mk("ok1"), mk("bad", mock={"fail": True}), mk("ok2")]
    events = asyncio.run(collect(cfgs))
    done = {e["model"] for e in by_model(events, "done")}
    errs = by_model(events, "error")
    assert done == {"ok1", "ok2"}
    assert len(errs) == 1 and errs[0]["model"] == "bad" and errs[0]["status"] == "error"
    assert events[-1]["event"] == "complete"


def test_rate_limited_status_is_distinct():
    events = asyncio.run(collect([mk("a"), mk("rl", mock={"rate_limited": True})]))
    assert by_model(events, "error")[0]["status"] == "rate_limited"


def test_first_token_timeout_and_total_timeout():
    cfgs = [mk("slow-start", first_token_timeout_s=0.2, timeout_s=5, mock={"hang": True}),
            mk("too-long", timeout_s=0.35, mock={"first_token_ms": 10, "tokens_per_s": 30}),
            mk("fine")]
    events = asyncio.run(collect(cfgs))
    errs = {e["model"]: e for e in by_model(events, "error")}
    assert errs["slow-start"]["status"] == "timeout" and "No response started" in errs["slow-start"]["error"]
    assert errs["too-long"]["status"] == "timeout" and "finish" in errs["too-long"]["error"]
    assert errs["too-long"]["chars"] > 0  # partial output is kept
    assert {e["model"] for e in by_model(events, "done")} == {"fine"}


def test_closing_the_stream_cancels_upstream_calls():
    finished = []
    cfg = mk("long", mock={"first_token_ms": 10, "tokens_per_s": 5, "on_finish": lambda: finished.append(1)})

    async def go():
        s = settings()
        gen = compare_events([cfg, mk("b")], Gateway(s), s, None, "p", {"temperature": 0.7, "max_tokens": 50}, "cmp_t")
        async for e in gen:
            if e["event"] == "delta":
                break  # the client "disconnects" mid-stream
        await gen.aclose()
        return len(asyncio.all_tasks())

    remaining = asyncio.run(go())
    assert finished, "upstream generator was not closed"
    assert remaining == 1  # only the test's own task is left


def test_tokens_are_per_model_and_cost_is_estimated():
    a = mk("a", price=(1.0, 2.0), mock={"first_token_ms": 5, "tokens_per_s": 1e5})
    b = mk("bbbbbbbb", price=(10.0, 20.0), mock={"first_token_ms": 5, "tokens_per_s": 1e5})
    events = asyncio.run(collect([a, b]))
    d = {e["model"]: e for e in by_model(events, "done")}
    assert d["a"]["input_tokens"] != d["bbbbbbbb"]["input_tokens"] or d["a"]["output_tokens"] != d["bbbbbbbb"]["output_tokens"]
    assert d["a"]["est_cost_usd"] == est_cost(a, d["a"]["input_tokens"], d["a"]["output_tokens"])
    assert d["bbbbbbbb"]["est_cost_usd"] > d["a"]["est_cost_usd"]
    assert d["a"]["tokens_approximate"] is False


def test_missing_pricing_gives_no_cost():
    assert est_cost(mk("x", price=(None, None)), 10, 10) is None


# ---------- DigitalOcean adapter against a fake HTTP server ----------

def sse_body(*objs):
    return "".join(f"data: {json.dumps(o)}\n\n" for o in objs) + "data: [DONE]\n\n"


def do_gateway(handler):
    s = settings(access_key="k")
    client = httpx.AsyncClient(base_url="https://x.test", transport=httpx.MockTransport(handler))
    return s, Gateway(s, do=DOAdapter(s, client))


def live_cfg(**kw):
    return ModelCfg(id="live-model", label="L", provider="t", context=1000, api="chat", input_per_m=1, output_per_m=2, **kw)


async def run_live(handler, cfg=None):
    s, gw = do_gateway(handler)
    out = []
    async for e in compare_events([cfg or live_cfg()], gw, s, "sys", "hi", {"temperature": 0.5, "max_tokens": 50}, "cmp_t"):
        out.append(e)
    return out


def test_do_adapter_parses_stream_usage_and_sends_expected_request():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        body = sse_body({"choices": [{"delta": {"role": "assistant"}}]},
                        {"choices": [{"delta": {"content": "Hel"}}]},
                        {"choices": [{"delta": {"content": "lo"}}]},
                        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3}})
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    events = asyncio.run(run_live(handler))
    done = by_model(events, "done")[0]
    assert "".join(e["text"] for e in by_model(events, "delta")) == "Hello"
    assert (done["input_tokens"], done["output_tokens"], done["tokens_approximate"]) == (12, 3, False)
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["stream"] is True and seen["body"]["stream_options"] == {"include_usage": True}
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}


def test_do_adapter_estimates_tokens_when_usage_missing():
    def handler(req):
        return httpx.Response(200, text=sse_body({"choices": [{"delta": {"content": "x" * 40}}]}))

    done = by_model(asyncio.run(run_live(handler)), "done")[0]
    assert done["tokens_approximate"] is True and done["output_tokens"] == 10


def test_do_adapter_retries_once_on_429_then_succeeds():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, text="slow down", headers={"retry-after": "0"})
        return httpx.Response(200, text=sse_body({"choices": [{"delta": {"content": "ok"}}]}))

    events = asyncio.run(run_live(handler))
    assert len(calls) == 2 and by_model(events, "done")


def test_do_adapter_gives_up_after_one_retry_and_classifies():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(429, text="no", headers={"retry-after": "0"})

    err = by_model(asyncio.run(run_live(handler)), "error")[0]
    assert len(calls) == 2 and err["status"] == "rate_limited"


def test_do_adapter_does_not_retry_client_errors():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(404, text="model not found")

    err = by_model(asyncio.run(run_live(handler)), "error")[0]
    assert len(calls) == 1 and err["status"] == "error" and "404" in err["error"]


def test_reasoning_only_chunks_reset_first_token_clock_and_use_param_name():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, text=sse_body({"choices": [{"delta": {"reasoning_content": "hmm"}}]},
                                                 {"choices": [{"delta": {"content": "answer"}}]}))

    events = asyncio.run(run_live(handler, live_cfg(max_tokens_param="max_completion_tokens")))
    assert any(e.get("status") == "thinking" for e in events)
    assert "max_completion_tokens" in seen["body"] and "max_tokens" not in seen["body"]
    assert by_model(events, "done")


# ---------- HTTP API ----------

def client(**kw):
    s = settings(mock_speed=25, **kw)
    return TestClient(create_app(s, Registry(), Gateway(s), Limiter(s))), s


def post(c, models, prompt="hi", **kw):
    return c.post("/api/v1/compare", json={"prompt": prompt, "models": models, **kw})


def parse_sse(text):
    out = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


def test_models_endpoint_in_mock_mode():
    c, _ = client()
    j = c.get("/api/v1/models").json()
    assert j["mode"] == "mock" and len(j["defaults"]) == 3 and j["limits"]["max_models"] == 4


def test_sse_stream_end_to_end():
    c, _ = client()
    r = post(c, ["mock-fast", "mock-flaky", "mock-balanced"])
    assert r.headers["content-type"].startswith("text/event-stream")
    names = [n for n, _ in parse_sse(r.text)]
    assert names[0] == "start" and names[-1] == "complete"
    assert names.count("done") == 2 and names.count("error") == 1


def test_non_streaming_fallback_returns_one_json_body():
    c, _ = client()
    j = post(c, ["mock-fast", "mock-balanced"], **{}).json() if False else \
        c.post("/api/v1/compare?stream=false", json={"prompt": "hi", "models": ["mock-fast", "mock-balanced"]}).json()
    assert set(j["results"]) == {"mock-fast", "mock-balanced"}
    assert j["results"]["mock-fast"]["status"] == "success" and j["results"]["mock-fast"]["text"]


@pytest.mark.parametrize("models,code", [
    (["mock-fast"], "model_count"),
    (["mock-fast", "mock-balanced", "mock-thorough", "mock-flaky", "mock-hang"], "model_count"),
    (["mock-fast", "gpt-evil"], "unknown_model"),
    (["mock-fast", "mock-fast"], "model_count"),  # duplicates collapse to one
])
def test_validation_of_model_selection(models, code):
    c, _ = client()
    r = post(c, models)
    assert r.status_code == 422 and r.json()["error"]["code"] == code


def test_prompt_validation():
    c, s = client()
    assert post(c, ["mock-fast", "mock-balanced"], prompt="   ").json()["error"]["code"] == "empty_prompt"
    long = post(c, ["mock-fast", "mock-balanced"], prompt="x" * (s.max_prompt_chars + 1))
    assert long.status_code == 422 and long.json()["error"]["code"] == "prompt_too_long"


def test_max_tokens_is_capped_server_side():
    seen = {}
    s = settings(mock_speed=25)
    gw = Gateway(s)
    orig = gw.stream
    gw.stream = lambda cfg, system, prompt, params: (seen.update(params), orig(cfg, system, prompt, params))[1]
    c = TestClient(create_app(s, Registry(), gw, Limiter(s)))
    post(c, ["mock-fast", "mock-balanced"], parameters={"temperature": 0.5, "max_tokens": 999999})
    assert seen["max_tokens"] == s.max_output_tokens_cap


def test_per_ip_rate_limit():
    c, _ = client(compares_per_hour=2)
    for _ in range(2):
        assert post(c, ["mock-fast", "mock-balanced"]).status_code == 200
    r = post(c, ["mock-fast", "mock-balanced"])
    assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited" and "retry-after" in r.headers
    other = c.post("/api/v1/compare", json={"prompt": "hi", "models": ["mock-fast", "mock-balanced"]},
                   headers={"x-forwarded-for": "9.9.9.9"})
    assert other.status_code == 200


def test_global_daily_spend_ceiling_disables_compare():
    c, s = client(daily_spend_ceiling_usd=0.0001)
    c.app.state.limiter.add_spend(0.001)
    r = post(c, ["mock-fast", "mock-balanced"])
    assert r.status_code == 503 and r.json()["error"]["code"] == "daily_budget_reached"


def test_spend_accumulates_from_completed_models():
    c, _ = client()
    post(c, ["mock-fast", "mock-balanced"])
    assert c.app.state.limiter.spend_usd > 0


def test_concurrency_slot_is_released_after_stream():
    c, _ = client(concurrent_per_ip=1)
    for _ in range(3):
        assert post(c, ["mock-fast", "mock-balanced"]).status_code == 200


def test_summary_endpoint_and_preference_marker():
    c, _ = client()
    r = c.post("/api/v1/summarize", json={"prompt": "p", "results": [{"model": "a", "text": "x"}, {"model": "b", "text": "y"}]})
    assert r.status_code == 200 and r.json()["generated_by_model"] is True and r.json()["summary"]
    assert c.post("/api/v1/preference", json={"request_id": "cmp_1", "model": "mock-fast"}).status_code == 204
    assert c.app.state.picks["mock-fast"] == 1


def test_credentials_never_appear_in_responses():
    s = settings(access_key="sk-secret-123")
    c = TestClient(create_app(s, Registry(), Gateway(s, do=DOAdapter(s, httpx.AsyncClient(
        base_url="https://x.test", transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")))))))
    body = c.get("/api/v1/models").text + post(c, ["deepseek-v4.1-flash", "llama-4-maverick"]).text
    assert "sk-secret-123" not in body
