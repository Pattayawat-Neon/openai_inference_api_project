from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_URL = os.environ.get("LLM_API_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
REQUEST_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "120"))
MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
RETRY_BACKOFF_S = float(os.environ.get("LLM_RETRY_BACKOFF_S", "2"))

OUTPUT_DIR = Path(os.environ.get("LLM_OUTPUT_DIR", "./output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("local_llm_client")

# Local servers rarely need a real key, but the SDK requires a non-empty string.
client = OpenAI(
    base_url=BASE_URL,
    api_key=os.environ.get("LLM_API_KEY", "not-needed"),
    timeout=REQUEST_TIMEOUT_S,
    max_retries=0,  # we implement our own retry loop below for full control
)


# --------------------------------------------------------------------------
# Result wrapper
# --------------------------------------------------------------------------

@dataclass
class LLMResult:
    ok: bool
    task: str
    data: Any = None
    error: Optional[str] = None
    raw_text: Optional[str] = None
    model: str = MODEL_NAME
    duration_s: float = 0.0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "task": self.task,
            "data": self.data,
            "error": self.error,
            "model": self.model,
            "duration_s": round(self.duration_s, 3),
            "meta": self.meta,
        }


# --------------------------------------------------------------------------
# Core call wrapper: timeout handling + bounded retries + error handling
# --------------------------------------------------------------------------

def _chat_completion_with_retry(
    *,
    messages: list[dict],
    task_name: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    response_format: Optional[dict] = None,
) -> LLMResult:
    """Calls the local OpenAI-compatible endpoint with retries + error handling."""
    last_error: Optional[str] = None
    start = time.perf_counter()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs: dict[str, Any] = dict(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if response_format:
                kwargs["response_format"] = response_format

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            duration = time.perf_counter() - start

            logger.info(
                "[%s] success on attempt %d/%d (%.2fs)",
                task_name, attempt, MAX_RETRIES, duration,
            )
            return LLMResult(
                ok=True,
                task=task_name,
                raw_text=content,
                duration_s=duration,
                meta={"attempt": attempt, "finish_reason": response.choices[0].finish_reason},
            )

        except APITimeoutError as e:
            last_error = f"Timeout after {REQUEST_TIMEOUT_S}s: {e}"
            logger.warning("[%s] attempt %d/%d timed out", task_name, attempt, MAX_RETRIES)

        except APIConnectionError as e:
            last_error = f"Connection error (is the server running at {BASE_URL}?): {e}"
            logger.warning("[%s] attempt %d/%d connection error: %s", task_name, attempt, MAX_RETRIES, e)

        except APIStatusError as e:
            # Non-2xx from the server (400, 500, etc). Retrying rarely helps for 4xx.
            last_error = f"Server returned status {e.status_code}: {e.message}"
            logger.error("[%s] attempt %d/%d status error: %s", task_name, attempt, MAX_RETRIES, last_error)
            if 400 <= e.status_code < 500:
                break  # client error - don't burn retries

        except APIError as e:
            last_error = f"API error: {e}"
            logger.error("[%s] attempt %d/%d API error: %s", task_name, attempt, MAX_RETRIES, e)

        except Exception as e:  # noqa: BLE001 - last line of defense for a client library
            last_error = f"Unexpected error: {e}"
            logger.exception("[%s] attempt %d/%d unexpected error", task_name, attempt, MAX_RETRIES)

        if attempt < MAX_RETRIES:
            sleep_s = RETRY_BACKOFF_S * attempt  # linear backoff
            time.sleep(sleep_s)

    duration = time.perf_counter() - start
    return LLMResult(ok=False, task=task_name, error=last_error, duration_s=duration)


def _save_output(result: LLMResult, prefix: str) -> Path:
    """Persists a result to ./output/<prefix>_<timestamp>.json"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"{prefix}_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info("Saved output -> %s", path)
    return path


# --------------------------------------------------------------------------
# Use case 1: Text Summarization
# --------------------------------------------------------------------------

def summarize_text(
    text: str,
    bullet_points: int = 5,
    save: bool = True,
) -> LLMResult:
    """Summarizes long text into concise bullet points."""
    system_prompt = (
        "You are a precise summarization engine. Summarize the user's text into "
        f"exactly {bullet_points} concise bullet points capturing only the most "
        "important facts. Use '- ' for each bullet. Do not add commentary, "
        "preamble, or a conclusion."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    result = _chat_completion_with_retry(
        messages=messages,
        task_name="summarization",
        temperature=0.2,
        max_tokens=512,
    )
    if result.ok:
        result.data = result.raw_text.strip()

    if save:
        _save_output(result, "summary")
    return result


# --------------------------------------------------------------------------
# Use case 2: Language Translation (EN <-> TH)
# --------------------------------------------------------------------------

def translate_text(
    text: str,
    target_lang: str = "th",
    save: bool = True,
) -> LLMResult:
    """Translates text. target_lang: 'th' (Thai) or 'en' (English)."""
    lang_map = {"th": "Thai", "en": "English"}
    if target_lang not in lang_map:
        return LLMResult(ok=False, task="translation", error=f"Unsupported target_lang '{target_lang}'. Use 'th' or 'en'.")

    target_name = lang_map[target_lang]
    system_prompt = (
        f"You are a professional translator. Translate the user's text into {target_name}. "
        "Preserve meaning, tone, and formatting. Return ONLY the translated text, "
        "with no explanations, notes, or quotation marks."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    result = _chat_completion_with_retry(
        messages=messages,
        task_name="translation",
        temperature=0.2,
        max_tokens=1024,
    )
    if result.ok:
        result.data = result.raw_text.strip()
        result.meta["target_lang"] = target_lang

    if save:
        _save_output(result, f"translation_{target_lang}")
    return result


# --------------------------------------------------------------------------
# Use case 3: Structured Data Extraction (text -> JSON schema)
# --------------------------------------------------------------------------

DEFAULT_INSIGHT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Short title describing the text"},
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "key_entities": {"type": "array", "items": {"type": "string"}},
        "key_insights": {"type": "array", "items": {"type": "string"}},
        "action_items": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "sentiment", "key_entities", "key_insights"],
}


def extract_structured_data(
    text: str,
    schema: Optional[dict] = None,
    save: bool = True,
) -> LLMResult:
    """
    Extracts key insights from text into a JSON object matching `schema`
    (JSON-Schema-like dict). Falls back to DEFAULT_INSIGHT_SCHEMA if none given.

    Uses the server's JSON mode (response_format={"type": "json_object"}),
    supported by vLLM's OpenAI-compatible server and modern Ollama builds.
    """
    schema = schema or DEFAULT_INSIGHT_SCHEMA

    system_prompt = (
        "You are a data extraction engine. Read the user's text and output a "
        "single JSON object that strictly matches this JSON Schema:\n\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        "Output ONLY valid JSON. No markdown fences, no commentary."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    result = _chat_completion_with_retry(
        messages=messages,
        task_name="structured_extraction",
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )

    if result.ok:
        raw = result.raw_text.strip()
        # Defensive cleanup in case the model wraps output in markdown fences
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0]
        try:
            result.data = json.loads(raw)
        except json.JSONDecodeError as e:
            result.ok = False
            result.error = f"Model did not return valid JSON: {e}"
            result.meta["raw_output"] = raw

    if save:
        _save_output(result, "extraction")
    return result


# --------------------------------------------------------------------------
# Demo / smoke test
# --------------------------------------------------------------------------

def _print_result(label: str, result: LLMResult) -> None:
    print(f"\n=== {label} ===")
    if result.ok:
        print(json.dumps(result.data, ensure_ascii=False, indent=2))
        print(f"(took {result.duration_s:.2f}s)")
    else:
        print(f"FAILED: {result.error}", file=sys.stderr)


if __name__ == "__main__":
    sample_text = (
        "The local inference server exposes an OpenAI-compatible API on port 8000. "
        "It runs entirely on-premise using a self-hosted GPU, so no data leaves the "
        "network. Teams can point any existing OpenAI SDK-based script at it by just "
        "changing the base_url. This reduces both latency and per-token cost compared "
        "to cloud APIs, while keeping sensitive documents private. The main trade-off "
        "is that the team is now responsible for GPU capacity planning and model updates."
    )

    _print_result("Summarization", summarize_text(sample_text, bullet_points=4))
    _print_result("Translation (EN -> TH)", translate_text(sample_text, target_lang="th"))
    _print_result("Structured Extraction", extract_structured_data(sample_text))

    print(f"\nAll results saved to: {OUTPUT_DIR.resolve()}")
