"""Shared MiniMax M2.7 client via OpenRouter (OpenAI-compatible API).

OpenRouter proxies a large pool of models behind a single OpenAI-compatible
endpoint, so we use the stock `openai` SDK pointed at openrouter.ai.

Model id: `minimax/minimax-m2.7`
Pricing: $0.30/M input, $1.20/M output (as of 2026-04).
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from dotenv import load_dotenv
from openai import APIStatusError, APITimeoutError, OpenAI, RateLimitError

load_dotenv()

_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "minimax/minimax-m2.7"
_API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("MINIMAX_API_KEY")

if not _API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in."
    )

# Optional attribution headers (OpenRouter uses these for rankings; safe to leave).
_EXTRA_HEADERS = {
    "HTTP-Referer": "https://github.com/gliner-az",
    "X-Title": "gliner-az",
}

_client = OpenAI(base_url=_BASE_URL, api_key=_API_KEY, default_headers=_EXTRA_HEADERS)


def call_claude(
    system: str,
    user: str,
    model: str = _DEFAULT_MODEL,
    max_tokens: int = 2048,
    max_retries: int = 4,
) -> Optional[str]:
    """Call MiniMax-M2.7 via OpenRouter with retry on transient errors. Returns text or None.

    Name preserved for backwards compatibility with existing callers, even though
    the underlying provider is MiniMax now.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    for attempt in range(max_retries):
        try:
            resp = _client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                # MiniMax-M2.7 is a reasoning model; we want plain text only.
                # Without this, tokens get consumed by hidden reasoning and
                # `content` comes back None when max_tokens is tight.
                # MiniMax-M2.7 is a reasoning model; OpenRouter won't let us
                # disable reasoning outright, but `exclude: true` hides the
                # reasoning trace from the response. `max_tokens` must still
                # cover reasoning + answer, so default generously (2048+).
                extra_body={"reasoning": {"exclude": True}},
            )
            if not resp.choices:
                return None
            content = resp.choices[0].message.content
            return content or None
        except (RateLimitError, APITimeoutError) as e:
            if attempt == max_retries - 1:
                print(f"[llm_client] giving up after {max_retries} retries: {e}")
                return None
            time.sleep(2 ** attempt)
        except APIStatusError as e:
            status = getattr(e, "status_code", 0)
            if 500 <= status < 600 and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"[llm_client] API error {status}: {e}")
            return None
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries - 1:
                print(f"[llm_client] unexpected error: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def call_claude_json(
    system: str,
    user: str,
    model: str = _DEFAULT_MODEL,
    max_tokens: int = 2048,
) -> Optional[dict]:
    """Call MiniMax expecting a JSON object. Strips markdown fences, parses, returns dict or None."""
    raw = call_claude(system, user, model=model, max_tokens=max_tokens)
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    # Reasoning budget is hidden but still consumes max_tokens on OpenRouter —
    # use a generous budget even for short smoke tests.
    reply = call_claude("You are a terse assistant.", "Say 'hello' in Azerbaijani.", max_tokens=512)
    print(reply)
