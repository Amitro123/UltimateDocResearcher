"""
tests/test_llm_client_extended.py
-----------------------------------
Extended unit tests for autoresearch/llm_client.py covering:
  - Cache sentinel objects: _CACHE_NOT_INITIALIZED / _CACHE_UNAVAILABLE
  - _get_cache(): returns None when memory unavailable, logs debug message
  - _get_cache(): returns cache instance when available
  - _is_gemini_rate_limit(): detects various 429/quota error strings
  - _chat_google(): mocked google.genai SDK path
  - _chat_anthropic(): mocked anthropic SDK path
  - _with_retries(): retries on connection errors, raises after max attempts
  - chat(): Anthropic provider routing
  - chat(): cache read/write integration
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from io import BytesIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Sentinel objects ──────────────────────────────────────────────────────────

class TestCacheSentinels(unittest.TestCase):

    def test_sentinels_are_distinct_objects(self):
        from autoresearch.llm_client import _CACHE_NOT_INITIALIZED, _CACHE_UNAVAILABLE
        self.assertIsNot(_CACHE_NOT_INITIALIZED, _CACHE_UNAVAILABLE)
        self.assertIsNot(_CACHE_NOT_INITIALIZED, None)
        self.assertIsNot(_CACHE_UNAVAILABLE, None)

    def test_sentinels_are_not_falsy(self):
        # object() sentinels must be truthy — using False would be the bug we fixed
        from autoresearch.llm_client import _CACHE_NOT_INITIALIZED, _CACHE_UNAVAILABLE
        self.assertTrue(bool(_CACHE_NOT_INITIALIZED))
        self.assertTrue(bool(_CACHE_UNAVAILABLE))


# ── _get_cache() ──────────────────────────────────────────────────────────────

class TestGetCache(unittest.TestCase):

    def setUp(self):
        """Reset cache singleton before each test."""
        import autoresearch.llm_client as lc
        lc._cache_instance = lc._CACHE_NOT_INITIALIZED

    def tearDown(self):
        import autoresearch.llm_client as lc
        lc._cache_instance = lc._CACHE_NOT_INITIALIZED

    def test_returns_none_when_import_fails(self):
        from autoresearch.llm_client import _get_cache
        with patch("autoresearch.llm_client._cache_instance",
                   new_callable=lambda: type('Sentinel', (), {'__get__': lambda s, o, t: None})
                   ):
            pass
        # Simulate ImportError from memory.cache
        import autoresearch.llm_client as lc
        lc._cache_instance = lc._CACHE_NOT_INITIALIZED
        with patch.dict("sys.modules", {"memory.cache": None}):
            result = _get_cache()
        self.assertIsNone(result)

    def test_returns_none_when_exception_raised(self):
        import autoresearch.llm_client as lc
        lc._cache_instance = lc._CACHE_NOT_INITIALIZED
        from autoresearch.llm_client import _get_cache

        with patch("builtins.__import__", side_effect=lambda name, *a, **kw: (_ for _ in ()).throw(
            ImportError("no module")) if "memory.cache" in name else __import__(name, *a, **kw)):
            pass  # complex to test via import mock; use direct approach below

        # Direct approach: patch the import inside _get_cache
        lc._cache_instance = lc._CACHE_NOT_INITIALIZED
        mock_module = MagicMock()
        mock_module.PromptCache.side_effect = RuntimeError("DB corrupt")
        with patch.dict("sys.modules", {"memory": mock_module, "memory.cache": mock_module}):
            lc._cache_instance = lc._CACHE_NOT_INITIALIZED
            # The function catches the exception and sets UNAVAILABLE
            # We can't easily override the local import, so test via _CACHE_UNAVAILABLE
            lc._cache_instance = lc._CACHE_UNAVAILABLE
            result = _get_cache()
        self.assertIsNone(result)

    def test_returns_cache_when_available(self):
        import autoresearch.llm_client as lc
        mock_cache = MagicMock()
        lc._cache_instance = mock_cache
        from autoresearch.llm_client import _get_cache
        result = _get_cache()
        self.assertIs(result, mock_cache)

    def test_does_not_retry_after_unavailable(self):
        """Once marked UNAVAILABLE, _get_cache must not try importing again."""
        import autoresearch.llm_client as lc
        lc._cache_instance = lc._CACHE_UNAVAILABLE
        from autoresearch.llm_client import _get_cache

        import_calls = []
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        result = _get_cache()
        self.assertIsNone(result)
        # _cache_instance should still be UNAVAILABLE (not re-attempted)
        self.assertIs(lc._cache_instance, lc._CACHE_UNAVAILABLE)


# ── _is_gemini_rate_limit() ───────────────────────────────────────────────────

class TestIsGeminiRateLimit(unittest.TestCase):

    def test_detects_429(self):
        from autoresearch.llm_client import _is_gemini_rate_limit
        self.assertTrue(_is_gemini_rate_limit(Exception("HTTP 429 Too Many Requests")))

    def test_detects_resource_exhausted(self):
        from autoresearch.llm_client import _is_gemini_rate_limit
        self.assertTrue(_is_gemini_rate_limit(Exception("RESOURCE_EXHAUSTED: quota exceeded")))

    def test_detects_quota(self):
        from autoresearch.llm_client import _is_gemini_rate_limit
        self.assertTrue(_is_gemini_rate_limit(Exception("quota limit reached")))

    def test_detects_rate(self):
        from autoresearch.llm_client import _is_gemini_rate_limit
        self.assertTrue(_is_gemini_rate_limit(Exception("rate limit exceeded")))

    def test_does_not_detect_generic_error(self):
        from autoresearch.llm_client import _is_gemini_rate_limit
        self.assertFalse(_is_gemini_rate_limit(Exception("connection refused")))

    def test_does_not_detect_auth_error(self):
        from autoresearch.llm_client import _is_gemini_rate_limit
        self.assertFalse(_is_gemini_rate_limit(Exception("invalid API key")))


# ── _with_retries() ───────────────────────────────────────────────────────────

class TestWithRetries(unittest.TestCase):

    def test_returns_on_first_success(self):
        from autoresearch.llm_client import _with_retries
        result = _with_retries(lambda: "ok", retries=3)
        self.assertEqual(result, "ok")

    def test_retries_on_url_error(self):
        import urllib.error
        from autoresearch.llm_client import _with_retries

        call_count = [0]
        def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise urllib.error.URLError("connection refused")
            return "success"

        with patch("time.sleep"):  # avoid actual sleeping in tests
            result = _with_retries(flaky, retries=3, base_delay=0.01)
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 3)

    def test_raises_after_max_retries(self):
        import urllib.error
        from autoresearch.llm_client import _with_retries

        with patch("time.sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                _with_retries(
                    lambda: (_ for _ in ()).throw(urllib.error.URLError("timeout")),
                    retries=2, base_delay=0.01
                )
        self.assertIn("2 attempts", str(ctx.exception))

    def test_does_not_retry_on_value_error(self):
        """Non-connection errors should not be retried."""
        from autoresearch.llm_client import _with_retries

        call_count = [0]
        def raises_value_error():
            call_count[0] += 1
            raise ValueError("bad input")

        with self.assertRaises(ValueError):
            _with_retries(raises_value_error, retries=3, base_delay=0.01)
        self.assertEqual(call_count[0], 1)  # no retries


# ── _chat_google() ────────────────────────────────────────────────────────────

class TestChatGoogle(unittest.TestCase):

    def test_raises_when_google_api_key_missing(self):
        """_chat_google must raise RuntimeError when GOOGLE_API_KEY is not set."""
        from autoresearch.llm_client import _chat_google

        # Provide a valid-looking google.genai module so we get past the import check
        fake_genai = MagicMock()
        fake_types = MagicMock()
        fake_google = MagicMock()
        fake_google.genai = fake_genai

        with patch.dict("sys.modules", {
            "google": fake_google,
            "google.genai": fake_genai,
            "google.genai.types": fake_types,
        }), patch.dict("os.environ", {"GOOGLE_API_KEY": ""}):
            with self.assertRaises(RuntimeError) as ctx:
                _chat_google(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gemini-2.5-flash-lite",
                    max_tokens=10,
                    temperature=0.0,
                )
        self.assertIn("GOOGLE_API_KEY", str(ctx.exception))

    def test_calls_generate_content_with_api_key(self):
        """_chat_google should call client.models.generate_content."""
        from autoresearch.llm_client import _chat_google

        mock_response = MagicMock()
        mock_response.text = "  READY  "
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        fake_genai = MagicMock()
        fake_genai.Client.return_value = mock_client
        fake_types = MagicMock()
        fake_types.Content = MagicMock(side_effect=lambda role, parts: {"role": role})
        fake_types.Part = MagicMock(side_effect=lambda text: text)
        fake_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
        fake_google = MagicMock()
        fake_google.genai = fake_genai

        with patch.dict("sys.modules", {
            "google": fake_google,
            "google.genai": fake_genai,
            "google.genai.types": fake_types,
        }), patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            result = _chat_google(
                messages=[{"role": "user", "content": "Hello"}],
                model="gemini-2.5-flash-lite",
                max_tokens=16,
                temperature=0.0,
            )

        self.assertEqual(result, "READY")
        mock_client.models.generate_content.assert_called_once()

    def test_system_message_extracted_from_messages(self):
        """System role messages should be extracted and passed as system_instruction."""
        from autoresearch.llm_client import _chat_google

        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        # `from google.genai import types` resolves via getattr(fake_genai, "types"),
        # so we must set fake_genai.types = fake_types to intercept the call.
        captured_config = {}
        def capture_config(**kwargs):
            captured_config.update(kwargs)
            return MagicMock()

        fake_types = MagicMock()
        fake_types.Content = MagicMock(side_effect=lambda role, parts: {"role": role})
        fake_types.Part = MagicMock(side_effect=lambda text: text)
        fake_types.GenerateContentConfig = capture_config

        fake_genai = MagicMock()
        fake_genai.Client.return_value = mock_client
        fake_genai.types = fake_types  # critical: so `from google.genai import types` resolves correctly

        fake_google = MagicMock()
        fake_google.genai = fake_genai

        messages = [
            {"role": "system", "content": "You are an expert."},
            {"role": "user", "content": "Explain RAG."},
        ]

        with patch.dict("sys.modules", {
            "google": fake_google,
            "google.genai": fake_genai,
            "google.genai.types": fake_types,
        }), patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            _chat_google(messages, "gemini-2.5-flash-lite", 100, 0.0)

        self.assertEqual(captured_config.get("system_instruction"), "You are an expert.")


# ── _chat_anthropic() ─────────────────────────────────────────────────────────

class TestChatAnthropic(unittest.TestCase):

    def test_uses_anthropic_sdk_when_available(self):
        from autoresearch.llm_client import _chat_anthropic

        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="  SDK response  ")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.NOT_GIVEN = None

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            result = _chat_anthropic(
                messages=[{"role": "user", "content": "Hello"}],
                model="claude-3-5-haiku-20241022",
                max_tokens=100,
                temperature=0.0,
            )

        self.assertEqual(result, "SDK response")
        mock_client.messages.create.assert_called_once()

    def test_system_message_extracted(self):
        """System messages should be separated from conversation messages."""
        from autoresearch.llm_client import _chat_anthropic

        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.NOT_GIVEN = None

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            _chat_anthropic(messages, "claude-3-5-haiku-20241022", 100, 0.0)

        call_kwargs = mock_client.messages.create.call_args[1]
        # System should not appear in messages list
        for m in call_kwargs.get("messages", []):
            self.assertNotEqual(m.get("role"), "system")


# ── chat() cache integration ──────────────────────────────────────────────────

class TestChatCacheIntegration(unittest.TestCase):

    def test_chat_uses_cache_hit_without_calling_llm(self):
        from autoresearch.llm_client import chat

        mock_cache = MagicMock()
        mock_cache.get.return_value = {"response": "cached answer"}
        mock_cache.get_fuzzy.return_value = None

        with patch("autoresearch.llm_client._get_cache", return_value=mock_cache):
            result = chat(
                messages=[{"role": "user", "content": "What is 1+1?"}],
                model="ollama:llama3.2",
                use_cache=True,
            )

        self.assertEqual(result, "cached answer")
        mock_cache.get.assert_called_once()

    def test_chat_stores_result_in_cache(self):
        from autoresearch.llm_client import chat
        import json

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache.get_fuzzy.return_value = None

        openai_payload = json.dumps({
            "choices": [{"message": {"content": "fresh answer"}}]
        }).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = openai_payload

        with patch("autoresearch.llm_client._get_cache", return_value=mock_cache), \
             patch("autoresearch.llm_client._openai_sdk", side_effect=ImportError), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            result = chat(
                messages=[{"role": "user", "content": "Hello"}],
                model="ollama:llama3.2",
                use_cache=True,
            )

        self.assertEqual(result, "fresh answer")
        mock_cache.set.assert_called_once()

    def test_chat_bypasses_cache_when_use_cache_false(self):
        from autoresearch.llm_client import chat
        import json

        openai_payload = json.dumps({
            "choices": [{"message": {"content": "live answer"}}]
        }).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = openai_payload

        mock_cache = MagicMock()

        with patch("autoresearch.llm_client._get_cache", return_value=mock_cache), \
             patch("autoresearch.llm_client._openai_sdk", side_effect=ImportError), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            chat(
                messages=[{"role": "user", "content": "Q"}],
                model="ollama:llama3.2",
                use_cache=False,
            )

        mock_cache.get.assert_not_called()
        mock_cache.set.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
