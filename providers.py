"""
Provider registry and LLM client wrappers for the dissent ABM.

Provenance tags: each provider carries a `regime` string ("chinese" / "western"
/ "open"). This is an internal provenance label only -- it records which
provider produced a row of output. It does NOT encode, and must not be read as,
a claim about national or political model behavior. The paper reports results by
model name and treats the comparison as model-specific (see Limitations).

Model IDs churn frequently; verify each is live on the run date and record that
date in the methods section.
"""

import os
import json
import time
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

try:
    from openai import OpenAI, AsyncOpenAI
    _OPENAI_SDK = True
except ImportError:
    _OPENAI_SDK = False


PROVIDERS = {
    "deepseek": {
        "regime":   "chinese",   # provenance tag only; see module docstring
        "label":    "DeepSeek V4 Flash (via OpenRouter)",
        "base_url": "https://openrouter.ai/api/v1",
        "model":    "deepseek/deepseek-v4-flash",
        "key_env":  "OPENROUTER_API_KEY",
    },
    "openai": {
        "regime":   "western",
        "label":    "GPT-4.1 mini (via OpenRouter)",
        "base_url": "https://openrouter.ai/api/v1",
        "model":    "openai/gpt-4.1-mini",
        "key_env":  "OPENROUTER_API_KEY",
    },
    "llama": {
        "regime":   "open",
        "label":    "Llama-4 Maverick (via OpenRouter)",
        "base_url": "https://openrouter.ai/api/v1",
        "model":    "meta-llama/llama-4-maverick",
        "key_env":  "OPENROUTER_API_KEY",
    },
}


@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    regime: str
    raw_ok: bool
    error: Optional[str] = None
    refusal_type: Optional[str] = None


class LLMProvider:
    """OpenAI-compatible client wrapper for one provider."""

    def __init__(self, provider_key: str):
        if provider_key not in PROVIDERS:
            raise ValueError(f"Unknown provider '{provider_key}'. Choices: {list(PROVIDERS)}")
        if not _OPENAI_SDK:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        cfg = PROVIDERS[provider_key]
        self.key_provider = provider_key
        self.regime = cfg["regime"]
        self.label = cfg["label"]
        self.model = cfg["model"]

        api_key = os.getenv(cfg["key_env"], "")
        if not api_key:
            raise RuntimeError(f"Missing API key: set {cfg['key_env']} for {cfg['label']}.")

        kwargs = {"api_key": api_key}
        if cfg["base_url"]:
            kwargs["base_url"] = cfg["base_url"]
        self.client = OpenAI(**kwargs)
        self.async_client = AsyncOpenAI(**kwargs)
        log.info(f"  Provider ready: {self.label} (regime={self.regime}, model={self.model})")

    def complete_json(self, prompt: str, temperature: float = 0.85,
                      max_tokens: int = 400, retries: int = 3) -> LLMResponse:
        last_err = None
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content or "{}"
                return LLMResponse(content=content, provider=self.key_provider,
                                   model=self.model, regime=self.regime, raw_ok=True)
            except Exception as e:
                last_err = str(e)
                if "response_format" in last_err.lower():
                    # Some providers reject response_format; retry asking for JSON in-prompt.
                    try:
                        resp = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user",
                                       "content": prompt + "\n\nReturn ONLY valid JSON."}],
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        content = resp.choices[0].message.content or "{}"
                        return LLMResponse(content=content, provider=self.key_provider,
                                           model=self.model, regime=self.regime, raw_ok=True)
                    except Exception as e2:
                        last_err = str(e2)
                time.sleep(1.5 * (attempt + 1))

        log.warning(f"  [{self.label}] call failed after {retries} tries: {last_err}")
        return LLMResponse(content="{}", provider=self.key_provider, model=self.model,
                           regime=self.regime, raw_ok=False, error=last_err)

    async def complete_json_async(self, prompt: str, temperature: float = 0.85,
                                  max_tokens: int = 400, retries: int = 4) -> LLMResponse:
        last_err = None
        for attempt in range(retries):
            try:
                resp = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content or "{}"
                return LLMResponse(content=content, provider=self.key_provider,
                                   model=self.model, regime=self.regime, raw_ok=True)
            except Exception as e:
                last_err = str(e)
                low = last_err.lower()
                if "response_format" in low:
                    try:
                        resp = await self.async_client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user",
                                       "content": prompt + "\n\nReturn ONLY valid JSON."}],
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        content = resp.choices[0].message.content or "{}"
                        return LLMResponse(content=content, provider=self.key_provider,
                                           model=self.model, regime=self.regime, raw_ok=True)
                    except Exception as e2:
                        last_err = str(e2)
                        low = last_err.lower()
                # Longer backoff on rate-limit/overload so concurrent bursts self-throttle.
                if any(k in low for k in ("rate", "429", "overload", "capacity", "too many")):
                    wait = 5.0 * (2 ** attempt)
                else:
                    wait = 1.5 * (attempt + 1)
                await asyncio.sleep(wait)

        log.warning(f"  [{self.label}] async call failed after {retries}: {last_err}")
        return LLMResponse(content="{}", provider=self.key_provider, model=self.model,
                           regime=self.regime, raw_ok=False, error=last_err)


def parse_json_safe(text: str) -> dict:
    """Parse JSON, tolerating markdown fences and surrounding prose."""
    if not text:
        return {}
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if "```" in t[3:] else t.lstrip("`")
        if t.startswith("json"):
            t = t[4:]
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        t = t[start:end + 1]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return {}


# Phrases that signal a content/safety refusal rather than a formatting slip.
_REFUSAL_MARKERS = (
    "i can't", "i cannot", "i'm unable", "i am unable", "i won't", "i will not",
    "as an ai", "i'm not able", "cannot assist", "can't help with",
    "i must decline", "not appropriate", "i'm sorry", "i apologize",
    "unable to provide", "against my", "cannot roleplay", "can't roleplay",
    "我不能", "我无法", "抱歉", "无法提供", "不能协助",
)


def classify_refusal(resp_content: str, raw_ok: bool) -> str:
    """Bin a failed/empty parse into 'api_error', 'content', or 'format'."""
    if not raw_ok:
        return "api_error"
    text = (resp_content or "").strip().lower()
    if text in ("", "{}"):
        return "api_error"
    if any(m in text for m in _REFUSAL_MARKERS):
        return "content"
    return "format"
