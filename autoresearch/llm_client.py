"""
autoresearch/llm_client.py
--------------------------
Unified LLM client for all autoresearch modules.

Supports three providers via a single model-string convention:

  Provider     Model string examples
  ---------    ---------------------------------------------------
  OpenAI       "gpt-4o-mini"  "gpt-4o"  "gpt-3.5-turbo"
  Anthropic    "claude-3-5-haiku-20241022"  "claude-opus-4-6"
  Ollama       "ollama:llama3.2"  "ollama:mistral"  "ollama:phi4"
               "ollama:llama3.2@http://remote-host:11434"
Gemini       "gemini-1.5-flash"  "gemini-1.5-pro"

Usage:
    from autoresearch.llm_client import chat, check_ollama, list_ollama_models

    response = chat(
        messages=[{"role": "user", "content": "Explain LoRA in one sentence."}],
        model="ollama:llama3.2",
        system="You are a concise research assistant.",
        max_tokens=256,
    )
    print(response)  # plain string

    # Check Ollama is reachable before running a pipeline
    if check_ollama():
        print(list_ollama_models())
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# ── Retry helper ─────────────────────────────────────────────────────────────

def _with_retries(fn, *, retries: int = 3, base_delay: float = 1.0, jitter: float = 0.3):
    """
    Call *fn()* up to *retries* times with exponential back-off.

    Retries on:
      - urllib.error.URLError  (connection refused, timeout, DNS failure)
      - OSError / ConnectionError

    Raises the last exception if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except (urllib.error.URLError, OSError, ConnectionError) as exc:
            last_exc = exc
            if attempt < retries - 1:
                delay = base_delay * (2 ** attempt) + (jitter * (0.5 - __import__("random").random()))
                delay = max(0.1, delay)
                print(
                    f"  ⚠️  LLM connection error (attempt {attempt + 1}/{retries}): {exc}\n"
                    f"     Retrying in {delay:.1f}s …",
                    file=sys.stderr,
                )
                time.sleep(delay)
    raise RuntimeError(
        f"LLM call failed after {retries} attempts. Last error: {last_exc}"
    ) from last_exc


# ── Structural output validator ───────────────────────────────────────────────

import re as _re

def _validate_markdown_fences(text: str) -> None:
    """
    Raise ValueError if *text* contains an unclosed Markdown code fence.

    A well-formed document has an even number of triple-backtick (```) markers:
    each opening fence (```python, ```bash, etc.) is closed by a bare ```.
    An odd count means the LLM was cut off mid-block — typically caused by
    API safety-filter truncation or hitting max_tokens mid-code-block.
    """
    fence_count = len(_re.findall(r"^```", text, _re.MULTILINE))
    if fence_count % 2 != 0:
        raise ValueError(
            f"Truncated output detected: {fence_count} code fence marker(s) found "
            f"(expected an even number). Probable API truncation or safety block."
        )


# ── Optional prompt cache ─────────────────────────────────────────────────────
# PromptCache is imported lazily so llm_client works even without the memory
# module.  Pass use_cache=False to bypass it entirely.
_CACHE_NOT_INITIALIZED = object()   # sentinel: "never attempted"
_CACHE_UNAVAILABLE = object()       # sentinel: "tried and failed"
_cache_instance = _CACHE_NOT_INITIALIZED

# Fuzzy cache guard: skip similarity-based lookup when the total prompt
# exceeds this length. With 40k-char corpus extracts the embedding is
# dominated by the corpus text, causing false-positive matches between
# prompts that share the same corpus but have different system prompts
# (e.g., ARCHITECTURE vs CODE_SUGGESTER on the same research run).
_FUZZY_CACHE_MAX_PROMPT_CHARS = 20_000

import logging as _logging
_logger = _logging.getLogger(__name__)


def _get_cache():
    """Return the shared PromptCache instance, or None if unavailable."""
    global _cache_instance
    if _cache_instance is _CACHE_NOT_INITIALIZED:
        try:
            from memory.cache import PromptCache  # noqa: PLC0415
            _cache_instance = PromptCache()
        except Exception as exc:
            _logger.debug("[llm_client] Cache unavailable: %s", exc)
            _cache_instance = _CACHE_UNAVAILABLE
    return _cache_instance if _cache_instance is not _CACHE_UNAVAILABLE else None


# ── Default Ollama endpoint ────────────────────────────────────────────────────

OLLAMA_DEFAULT_BASE = "http://localhost:11434"
OLLAMA_V1_BASE = f"{OLLAMA_DEFAULT_BASE}/v1"   # OpenAI-compatible endpoint


# ── Model spec parsing ─────────────────────────────────────────────────────────

def parse_model(model: str) -> tuple[str, str, Optional[str]]:
    """
    Parse a model string into (provider, model_name, base_url).

    Examples:
        "gpt-4o-mini"                        → ("openai",    "gpt-4o-mini",   None)
        "claude-3-5-haiku-20241022"          → ("anthropic", "claude-3-5-...", None)
        "ollama:llama3.2"                    → ("ollama",    "llama3.2",      "http://localhost:11434")
        "ollama:mistral@http://host:11434"   → ("ollama",    "mistral",       "http://host:11434")
    """
    if model.startswith("ollama:"):
        rest = model[len("ollama:"):]
        if "@" in rest:
            model_name, base = rest.split("@", 1)
        else:
            model_name, base = rest, OLLAMA_DEFAULT_BASE
        return "ollama", model_name.strip(), base.rstrip("/")

    if model.startswith("claude"):
        return "anthropic", model, None

    if model.startswith("gemini"):
        return "google", model, None

    # Default: OpenAI-compatible
    return "openai", model, None


def is_ollama(model: str) -> bool:
    return model.startswith("ollama:")


# ── Health check ───────────────────────────────────────────────────────────────

def check_ollama(base_url: str = OLLAMA_DEFAULT_BASE, timeout: int = 3) -> bool:
    """
    Return True if an Ollama server is reachable at base_url.

    Quick ping to /api/tags — does not require the openai package.
    """
    try:
        req = urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout)
        return req.status == 200
    except Exception:
        return False


def list_ollama_models(base_url: str = OLLAMA_DEFAULT_BASE) -> list[str]:
    """
    Return the list of model names currently pulled on the local Ollama server.
    Returns an empty list if Ollama is not reachable.
    """
    try:
        req = urllib.request.urlopen(f"{base_url}/api/tags", timeout=5)
        data = json.loads(req.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ── Core chat function ────────────────────────────────────────────────────────

def chat(
    messages: list[dict],
    model: str = "ollama:llama3.2",
    system: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    api_base: Optional[str] = None,   # explicit override (skips parse_model)
    use_cache: bool = True,           # set False to always call the LLM
    cache_fuzzy_threshold: float = 0.92,  # similarity threshold for fuzzy hits
) -> str:
    """
    Send a chat request and return the assistant's reply as a plain string.

    Args:
        messages:              Chat messages in OpenAI format [{"role": ..., "content": ...}]
        model:                 Model string (see parse_model for format)
        system:                Optional system prompt (prepended as {"role": "system", ...})
        max_tokens:            Max completion tokens
        temperature:           Sampling temperature (0 = deterministic)
        api_base:              Override the inferred API base URL
        use_cache:             Whether to check/populate the PromptCache (default True)
        cache_fuzzy_threshold: Cosine-similarity threshold for fuzzy cache hits (0–1)

    Returns:
        The assistant reply text, or raises RuntimeError on failure.
    """
    if system:
        messages = [{"role": "system", "content": system}] + list(messages)

    # ── Cache lookup ──────────────────────────────────────────────────────────
    # Build a single prompt string for cache key purposes.
    cache_prompt = "\n".join(
        f"[{m['role']}] {m['content']}" for m in messages
    )
    cache = _get_cache() if use_cache else None

    if cache is not None:
        # 1. Exact hit (SHA1 match)
        entry = cache.get(cache_prompt, model=model)
        if entry is not None:
            return entry["response"]
        # 2. Fuzzy hit — only for short prompts. Large-corpus prompts (> 20k chars)
        # produce embeddings dominated by the corpus text, causing false-positive
        # matches across calls that share the same corpus but use different system
        # prompts (e.g., ARCHITECTURE vs CODE generation on the same research run).
        if len(cache_prompt) <= _FUZZY_CACHE_MAX_PROMPT_CHARS:
            entry = cache.get_fuzzy(cache_prompt, threshold=cache_fuzzy_threshold, model=model)
            if entry is not None:
                return entry["response"]

    # ── Real LLM call with structural validation + retry ──────────────────────
    provider, model_name, inferred_base = parse_model(model)
    effective_base = api_base or inferred_base

    _MAX_STRUCT_RETRIES = 2
    reply = ""
    for struct_attempt in range(_MAX_STRUCT_RETRIES + 1):
        if provider == "anthropic":
            reply = _chat_anthropic(messages, model_name, max_tokens, temperature)
        elif provider == "google":
            reply = _chat_google(messages, model_name, max_tokens, temperature)
        elif provider == "ollama":
            reply = _chat_openai_compat(
                messages, model_name, max_tokens, temperature,
                base_url=f"{effective_base}/v1",
                api_key="ollama",
            )
        else:
            # Default: OpenAI
            reply = _chat_openai_compat(
                messages, model_name, max_tokens, temperature,
                base_url=effective_base,
                api_key=os.getenv("OPENAI_API_KEY", ""),
            )

        try:
            _validate_markdown_fences(reply)
            break  # output is structurally valid
        except ValueError as ve:
            print(
                f"  ⚠️  Structural validation failed "
                f"(attempt {struct_attempt + 1}/{_MAX_STRUCT_RETRIES + 1}): {ve}",
                file=sys.stderr,
            )
            # Flush the cache entry so the broken response is never served again
            if cache is not None:
                try:
                    cache.invalidate(cache_prompt, model=model)
                except Exception:
                    pass
            if struct_attempt == _MAX_STRUCT_RETRIES:
                raise RuntimeError(
                    f"LLM output failed structural validation after "
                    f"{_MAX_STRUCT_RETRIES + 1} attempt(s): {ve}"
                ) from ve
            print("     Retrying LLM call …", file=sys.stderr)

    # ── Cache store ───────────────────────────────────────────────────────────
    if cache is not None:
        try:
            cache.set(cache_prompt, reply, model=model)
        except Exception:
            pass  # never let cache writes break the caller

    return reply


# ── Provider implementations ──────────────────────────────────────────────────

def _chat_openai_compat(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    base_url: Optional[str],
    api_key: str,
) -> str:
    """
    OpenAI-compatible chat completion.
    Works for: OpenAI API, Ollama /v1, LM Studio, vLLM, etc.
    Tries openai SDK first, falls back to raw urllib.
    """
    try:
        return _openai_sdk(messages, model, max_tokens, temperature, base_url, api_key)
    except ImportError:
        # openai package not installed — use raw HTTP (works for Ollama)
        return _urllib_openai(messages, model, max_tokens, temperature, base_url, api_key)


def _openai_sdk(
    messages, model, max_tokens, temperature, base_url, api_key
) -> str:
    from openai import OpenAI
    kwargs = {"api_key": api_key or "dummy"}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def _urllib_openai(
    messages, model, max_tokens, temperature, base_url, api_key,
    retries: int = 3,
) -> str:
    """
    Pure stdlib HTTP call to any OpenAI-compatible endpoint.
    No external packages required — important for Ollama when openai isn't installed.
    Retries up to *retries* times with exponential back-off on connection errors.
    """
    url = f"{base_url}/chat/completions" if base_url else "https://api.openai.com/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key or 'dummy'}",
    }

    def _do_request():
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(f"HTTP {e.code}: {body}") from e

    return _with_retries(_do_request, retries=retries)


def _chat_anthropic(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    # Separate system message from the conversation
    system = ""
    filtered = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            filtered.append(m)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or anthropic.NOT_GIVEN,
            messages=filtered,
            temperature=temperature,
        )
        return resp.content[0].text.strip()
    except ImportError:
        # anthropic SDK not installed — try urllib against public API
        return _urllib_anthropic(filtered, model, system, max_tokens, temperature)


def _urllib_anthropic(messages, model, system, max_tokens, temperature,
                       retries: int = 3) -> str:
    """Pure stdlib fallback for Anthropic API (retries on connection errors)."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set and anthropic SDK not installed")
    payload = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }).encode()
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    def _do_request():
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                return data["content"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(f"Anthropic HTTP {e.code}: {body}") from e

    return _with_retries(_do_request, retries=retries)


def _is_gemini_rate_limit(exc: Exception) -> bool:
    """Return True if *exc* looks like a Gemini 429 / quota error."""
    msg = str(exc).lower()
    return (
        "429" in str(exc)
        or "resource_exhausted" in msg
        or "quota" in msg
        or "rate" in msg
    )


def _chat_google(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    retries: int = 4,
    base_delay: float = 15.0,   # free-tier needs longer gaps than typical APIs
) -> str:
    """
    Google Gemini implementation via the google-genai SDK.

    Automatically retries on 429 / ResourceExhausted errors with exponential
    back-off.  Free-tier keys hit per-minute quotas quickly, so the default
    base_delay is 15 s (doubles each attempt: 15 → 30 → 60 → 120 s).
    """
    import random

    system = ""
    filtered = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            filtered.append(m)

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        raise RuntimeError(
            "google-genai SDK not installed. Run: pip install google-genai"
        )

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    client = genai.Client(api_key=api_key)

    # Build contents list from conversation history
    contents = []
    for m in filtered[:-1]:
        role = "user" if m["role"] == "user" else "model"
        contents.append(
            genai_types.Content(role=role, parts=[genai_types.Part(text=m["content"])])
        )
    last_msg = filtered[-1]["content"] if filtered else ""
    contents.append(
        genai_types.Content(role="user", parts=[genai_types.Part(text=last_msg)])
    )

    # Disable all safety filters: code generation (os, Path, subprocess, etc.)
    # frequently triggers false-positive HARM_CATEGORY_DANGEROUS_CONTENT blocks,
    # which silently truncate the response mid-generation.
    _BLOCK_NONE_CATEGORIES = [
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    ]
    cfg = genai_types.GenerateContentConfig(
        system_instruction=system if system else None,
        max_output_tokens=max_tokens,
        temperature=temperature,
        safety_settings=[
            genai_types.SafetySetting(category=c, threshold="BLOCK_NONE")
            for c in _BLOCK_NONE_CATEGORIES
        ],
    )

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=cfg,
            )
            return resp.text.strip()
        except Exception as exc:
            if _is_gemini_rate_limit(exc) and attempt < retries - 1:
                last_exc = exc
                delay = base_delay * (2 ** attempt) + random.uniform(0, 5)
                print(
                    f"  WARNING: Gemini rate limit (attempt {attempt + 1}/{retries}): {exc}\n"
                    f"     Retrying in {delay:.0f}s ...",
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                raise RuntimeError(f"Gemini API call failed: {exc}") from exc

    raise RuntimeError(
        f"Gemini API call failed after {retries} attempts. Last error: {last_exc}"
    ) from last_exc


# ── Convenience: detect best available model ──────────────────────────────────

def best_available_model(prefer_ollama_model: str = "llama3.2") -> str:
    """
    Auto-detect the best available model in order of preference:
      1. Ollama (free, local) — if server is reachable and model is pulled
      2. Anthropic Claude Haiku — if ANTHROPIC_API_KEY is set
      3. OpenAI gpt-4o-mini   — if OPENAI_API_KEY is set
      4. Ollama fallback       — returns the spec even if server not confirmed running

    Useful for running the pipeline without worrying about which key is set.
    """
    if check_ollama():
        models = list_ollama_models()
        if models:
            # Prefer the requested model; fall back to whatever is pulled
            target = next((m for m in models if prefer_ollama_model in m), models[0])
            return f"ollama:{target}"

    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude-3-5-haiku-20241022"

    if os.getenv("GOOGLE_API_KEY"):
        # Prioritize 2.5-flash-lite as requested, then 1.5, then 2.0
        return "gemini-2.5-flash-lite"

    if os.getenv("OPENAI_API_KEY"):
        return "gpt-4o-mini"

    # Last resort: return Ollama spec (will fail with helpful error if not running)
    return f"ollama:{prefer_ollama_model}"


# ── CLI (quick test / model check) ───────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    # Windows CP1252 terminals can't encode emoji — ensure UTF-8 output
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    import argparse

    parser = argparse.ArgumentParser(description="Test LLM client or check available models")
    subparsers = parser.add_subparsers(dest="cmd")

    # check subcommand
    check_p = subparsers.add_parser("check", help="Check Ollama / list models")
    check_p.add_argument("--base", default=OLLAMA_DEFAULT_BASE)

    # chat subcommand
    chat_p = subparsers.add_parser("chat", help="Send a test prompt")
    chat_p.add_argument("--model", default="ollama:llama3.2")
    chat_p.add_argument("--prompt", default="Explain LoRA fine-tuning in one sentence.")
    chat_p.add_argument("--max-tokens", type=int, default=256)

    # auto subcommand
    subparsers.add_parser("auto", help="Print best available model")

    args = parser.parse_args()

    if args.cmd == "check":
        ok = check_ollama(args.base)
        if ok:
            models = list_ollama_models(args.base)
            print(f"✅ Ollama reachable at {args.base}")
            print(f"   Models pulled: {', '.join(models) if models else '(none)'}")
            if not models:
                print("\n   Pull a model with:  ollama pull llama3.2")
            else:
                # Send a small "hello world" chat to confirm the model actually responds
                test_model = f"ollama:{models[0]}"
                print(f"\n   Sending hello-world chat to {test_model} …")
                try:
                    reply = chat(
                        messages=[{"role": "user", "content": "Reply with exactly the word: READY"}],
                        model=test_model,
                        max_tokens=16,
                        use_cache=False,
                    )
                    print(f"   ✅ Model responded: {reply!r}")
                except Exception as e:
                    print(f"   ❌ Model did not respond: {e}")
                    print("      The server is reachable but the model may be loading or unresponsive.")
                    print("      Try:  ollama run " + models[0])
        else:
            print(f"❌ Ollama not reachable at {args.base}")
            print("   Install: https://ollama.com/download")
            print("   Then:    ollama pull llama3.2")

    elif args.cmd == "chat":
        provider, model_name, _ = parse_model(args.model)
        print(f"Provider: {provider}  Model: {model_name}")
        try:
            reply = chat(
                messages=[{"role": "user", "content": args.prompt}],
                model=args.model,
                max_tokens=args.max_tokens,
            )
            print(f"\nResponse:\n{reply}")
        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.cmd == "auto":
        model = best_available_model()
        print(f"Best available model: {model}")

    else:
        parser.print_help()
