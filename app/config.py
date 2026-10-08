"""Settings and the model registry. Everything tunable lives here or in models.json."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).parent


@dataclass(frozen=True)
class ModelCfg:
    id: str
    label: str
    provider: str
    context: int
    api: str  # "chat" (OpenAI chat completions) | "mock"
    input_per_m: float | None = None
    output_per_m: float | None = None
    max_tokens_param: str = "max_tokens"
    timeout_s: float | None = None
    first_token_timeout_s: float | None = None
    mock: dict = field(default_factory=dict)

    def public(self, as_of: str) -> dict:
        return {
            "id": self.id, "label": self.label, "provider": self.provider,
            "context": self.context, "input_per_m": self.input_per_m,
            "output_per_m": self.output_per_m, "pricing_as_of": as_of,
        }


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass
class Settings:
    access_key: str | None = field(default_factory=lambda: os.environ.get("DO_MODEL_ACCESS_KEY") or None)
    base_url: str = field(default_factory=lambda: os.environ.get("DO_INFERENCE_BASE_URL", "https://inference.do-ai.run"))
    mock_speed: float = field(default_factory=lambda: _env_float("MOCK_SPEED", 1))
    force_mock: bool = field(default_factory=lambda: os.environ.get("MOCK_MODE") == "1")
    min_models: int = 2
    max_models: int = field(default_factory=lambda: _env_int("MAX_MODELS", 4))
    max_prompt_chars: int = field(default_factory=lambda: _env_int("MAX_PROMPT_CHARS", 8000))
    max_output_tokens_cap: int = field(default_factory=lambda: _env_int("MAX_OUTPUT_TOKENS_CAP", 2000))
    default_timeout_s: float = field(default_factory=lambda: _env_float("MODEL_TIMEOUT_S", 30))
    default_first_token_timeout_s: float = field(default_factory=lambda: _env_float("FIRST_TOKEN_TIMEOUT_S", 15))
    compares_per_hour: int = field(default_factory=lambda: _env_int("COMPARES_PER_HOUR", 10))
    concurrent_per_ip: int = field(default_factory=lambda: _env_int("CONCURRENT_PER_IP", 2))
    daily_spend_ceiling_usd: float = field(default_factory=lambda: _env_float("DAILY_SPEND_CEILING_USD", 5.0))

    @property
    def mock_mode(self) -> bool:
        return self.force_mock or not self.access_key


class Registry:
    def __init__(self, path: Path | None = None):
        raw = json.loads((path or _HERE / "models.json").read_text())
        self.as_of: str = raw["pricing_as_of"]
        self.source: str = raw["pricing_source"]
        self.default_ids: list[str] = raw["default_models"]
        self.summary_model: str = raw["summary_model"]
        self.live = {m["id"]: ModelCfg(**m) for m in raw["models"]}
        self.mock = {m["id"]: ModelCfg(**m) for m in raw["mock_models"]}

    def for_mode(self, mock_mode: bool) -> dict[str, ModelCfg]:
        return self.mock if mock_mode else self.live
