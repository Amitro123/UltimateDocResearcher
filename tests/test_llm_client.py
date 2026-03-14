"""
tests/test_llm_client.py
------------------------
Unit tests for autoresearch/llm_client.py.

Tests cover:
  - parse_model: all three provider formats + remote Ollama
  - is_ollama helper
  - check_ollama: graceful False when server unreachable
  - list_ollama_models: graceful [] when server unreachable
  - best_available_model: falls back through the priority chain
  - _urllib_openai: correct JSON payload structure
  - chat: routes to correct provider, heuristic fallback on failure
  - Ollama end-to-end path via mocked urllib (no server required)
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from io import BytesIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autoresearch.llm_client import (
    parse_model,
    is_ollama,
    check_ollama,
    list_ollama_models,
    best_available_model,
    OLLAMA_DEFAULT_BASE,
    OLLAMA_V1_BASE,
)


# ── parse_model ───────────────────────────────────────────────────────────────

class TestParseModel(unittest.TestCase):

    def test_openai_model(self):
        provider, name, base = parse_model("gpt-4o-mini")
        self.assertEqual(provider, "openai")
        self.assertEqual(name, "gpt-4o-mini")
        self.assertIsNone(base)

    def test_openai_other(self):
        provider, name, base = parse_model("gpt-3.5-turbo")
        self.assertEqual(provider, "openai")

    def test_anthropic_model(self):
        provider, name, base = parse_model("claude-3-5-haiku-20241022")
        self.assertEqual(provider, "anthropic")
        self.assertEqual(name, "claude-3-5-haiku-20241022")
        self.assertIsNone(base)

    def test_anthropic_opus(self):
        provider, name, _ = parse_model("claude-opus-4-6")
        self.assertEqual(provider, "anthropic")

    def test_ollama_simple(self):
        provider, name, base = parse_model("ollama:llama3.2")
        self.assertEqual(provider, "ollama")
        self.assertEqual(name, "llama3.2")
        self.assertEqual(base, OLLAMA_DEFAULT_BASE)

    def test_ollama_with_remote_host(self):
        provider, name, base = parse_model("ollama:mistral@http://remote:11434")
        self.assertEqual(provider, "ollama")
        self.assertEqual(name, "mistral")
        self.assertEqual(base, "http://remote:11434")

    def test_ollama_trailing_slash_stripped(self):
        _, _, base = parse_model("ollama:phi4@http://host:11434/")
        self.assertEqual(base, "http://host:11434")

    def test_ollama_model_with_tag(self):
        _, name, _ = parse_model("ollama:llama3.2:latest")
        self.assertEqual(name, "llama3.2:latest")


# ── is_ollama ─────────────────────────────────────────────────────────────────

class TestIsOllama(unittest.TestCase):

    def test_true_for_ollama_prefix(self):
        self.assertTrue(is_ollama("ollama:llama3.2"))

    def test_false_for_openai(self):
        self.assertFalse(is_ollama("gpt-4o-mini"))

    def test_false_for_anthropic(self):
        self.assertFalse(is_ollama("claude-3-5-haiku-20241022"))


# ── check_ollama ──────────────────────────────────────────────────────────────

class TestCheckOllama(unittest.TestCase):

    def test_returns_false_when_unreachable(self):
        # No Ollama server in CI/test env — should return False gracefully
        result = check_ollama(base_url="http://127.0.0.1:19999", timeout=1)
        self.assertFalse(result)

    def test_returns_false_on_invalid_host(self):
        result = check_ollama(base_url="http://nonexistent-host-xyz.local:11434", timeout=1)
        self.assertFalse(result)

    def test_returns_true_when_server_responds(self):
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.status = 200
        with patch("urllib.request.urlopen", return_value=mock_response):
            result = check_ollama()
        self.assertTrue(result)


# ── list_ollama_models ────────────────────────────────────────────────────────

class TestListOllamaModels(unittest.TestCase):

    def test_returns_empty_list_when_unreachable(self):
        models = list_ollama_models(base_url="http://127.0.0.1:19999")
        self.assertEqual(models, [])

    def test_parses_model_names_from_response(self):
        payload = json.dumps({
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "mistral:7b"},
            ]
        }).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = payload
        with patch("urllib.request.urlopen", return_value=mock_resp):
            models = list_ollama_models()
        self.assertEqual(models, ["llama3.2:latest", "mistral:7b"])

    def test_returns_empty_on_missing_models_key(self):
        payload = json.dumps({}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = payload
        with patch("urllib.request.urlopen", return_value=mock_resp):
            models = list_ollama_models()
        self.assertEqual(models, [])


# ── best_available_model ──────────────────────────────────────────────────────

class TestBestAvailableModel(unittest.TestCase):

    def test_prefers_ollama_when_running_with_model_pulled(self):
        with patch("autoresearch.llm_client.check_ollama", return_value=True), \
             patch("autoresearch.llm_client.list_ollama_models", return_value=["llama3.2:latest"]):
            model = best_available_model(prefer_ollama_model="llama3.2")
        self.assertTrue(model.startswith("ollama:"))
        self.assertIn("llama3.2", model)

    def test_falls_back_to_anthropic_when_ollama_down(self):
        with patch("autoresearch.llm_client.check_ollama", return_value=False), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test", "OPENAI_API_KEY": ""}):
            model = best_available_model()
        self.assertIn("claude", model)

    def test_falls_back_to_openai_when_no_anthropic(self):
        with patch("autoresearch.llm_client.check_ollama", return_value=False), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "sk-test"}):
            model = best_available_model()
        self.assertIn("gpt", model)

    def test_returns_ollama_spec_as_last_resort(self):
        with patch("autoresearch.llm_client.check_ollama", return_value=False), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""}):
            model = best_available_model(prefer_ollama_model="phi4")
        self.assertEqual(model, "ollama:phi4")

    def test_uses_first_available_model_when_preferred_not_pulled(self):
        with patch("autoresearch.llm_client.check_ollama", return_value=True), \
             patch("autoresearch.llm_client.list_ollama_models", return_value=["mistral:7b"]):
            model = best_available_model(prefer_ollama_model="llama3.2")
        # Should use mistral since llama3.2 isn't pulled
        self.assertIn("mistral", model)


# ── _urllib_openai (Ollama raw HTTP path) ─────────────────────────────────────

class TestUrllibOpenaiPath(unittest.TestCase):

    def _mock_response(self, content: str):
        payload = json.dumps({
            "choices": [{"message": {"content": content}}]
        }).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = payload
        return mock_resp

    def test_sends_correct_payload_to_ollama(self):
        from autoresearch.llm_client import _urllib_openai
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode())
            captured["auth"] = req.get_header("Authorization")
            return self._mock_response("test reply")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = _urllib_openai(
                messages=[{"role": "user", "content": "Hello"}],
                model="llama3.2",
                max_tokens=100,
                temperature=0.0,
                base_url=OLLAMA_V1_BASE,
                api_key="ollama",
            )

        self.assertEqual(result, "test reply")
        self.assertIn("/chat/completions", captured["url"])
        self.assertEqual(captured["body"]["model"], "llama3.2")
        self.assertEqual(captured["body"]["messages"][0]["content"], "Hello")
        self.assertIn("Bearer ollama", captured["auth"])

    def test_raises_on_http_error(self):
        import urllib.error
        from autoresearch.llm_client import _urllib_openai

        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
            url="http://x", code=404, msg="Not Found", hdrs=None, fp=BytesIO(b"model not found")
        )):
            with self.assertRaises(RuntimeError) as ctx:
                _urllib_openai([], "bad-model", 100, 0.0, OLLAMA_V1_BASE, "ollama")
        self.assertIn("404", str(ctx.exception))


# ── chat() integration via mocked urllib ──────────────────────────────────────

class TestChatFunction(unittest.TestCase):

    def _openai_mock(self, reply: str):
        payload = json.dumps({
            "choices": [{"message": {"content": reply}}]
        }).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = payload
        return mock_resp

    def test_ollama_chat_returns_reply(self):
        from autoresearch.llm_client import chat
        with patch("urllib.request.urlopen", return_value=self._openai_mock("  hello world  ")):
            result = chat(
                messages=[{"role": "user", "content": "Say hi"}],
                model="ollama:llama3.2",
            )
        self.assertEqual(result, "hello world")

    def test_system_prompt_prepended(self):
        from autoresearch.llm_client import chat
        captured_body = {}

        def fake_urlopen(req, timeout=None):
            captured_body.update(json.loads(req.data.decode()))
            return self._openai_mock("ok")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            chat(
                messages=[{"role": "user", "content": "Q"}],
                model="ollama:llama3.2",
                system="You are helpful.",
            )

        msgs = captured_body.get("messages", [])
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "You are helpful.")
        self.assertEqual(msgs[1]["role"], "user")

    def test_api_base_override_respected(self):
        from autoresearch.llm_client import chat
        captured_url = {}

        def fake_urlopen(req, timeout=None):
            captured_url["url"] = req.full_url
            return self._openai_mock("ok")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            chat(
                messages=[{"role": "user", "content": "Q"}],
                model="ollama:llama3.2",
                api_base="http://custom-host:11434",
            )

        self.assertIn("custom-host", captured_url["url"])

    def test_openai_model_uses_openai_key(self):
        from autoresearch.llm_client import chat
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["auth"] = req.get_header("Authorization")
            return self._openai_mock("response")

        import os
        with patch("urllib.request.urlopen", side_effect=fake_urlopen), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "sk-real-key"}):
            # Force SDK import to fail so urllib path is used
            with patch.dict(sys.modules, {"openai": None}):
                chat(
                    messages=[{"role": "user", "content": "Q"}],
                    model="gpt-4o-mini",
                )

        self.assertIn("sk-real-key", captured.get("auth", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
