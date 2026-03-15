"""
tests/test_scraper.py
---------------------
Unit tests for collector/scraper.py.

Tests cover:
  - _html_to_text() with and without BeautifulSoup
  - scrape_topic() safe to call from a synchronous context (no RuntimeError)
  - scrape_topic() dispatches to a thread pool when an event loop is already running
  - scrape_topic() returns a string (may be empty if no APIs configured)
"""
import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.scraper import _html_to_text, scrape_topic


# ── _html_to_text ─────────────────────────────────────────────────────────────

class TestHtmlToText(unittest.TestCase):

    def test_strips_html_tags_without_bs4(self):
        """Regex fallback removes tags when bs4 is absent."""
        html = "<p>Hello <b>world</b></p>"
        with patch("collector.scraper.HAS_BS4", False):
            result = _html_to_text(html)
        self.assertIn("Hello", result)
        self.assertNotIn("<p>", result)
        self.assertNotIn("<b>", result)

    def test_returns_string(self):
        result = _html_to_text("<html><body>content</body></html>")
        self.assertIsInstance(result, str)

    def test_empty_html_returns_string(self):
        result = _html_to_text("")
        self.assertIsInstance(result, str)


# ── scrape_topic() sync wrapper ───────────────────────────────────────────────

class TestScrapeTopicSync(unittest.TestCase):
    """
    scrape_topic() must not raise RuntimeError regardless of whether an
    event loop is already running (Jupyter / FastAPI scenario — the asyncio fix).
    """

    def _make_mock_session(self):
        """Return a mock aiohttp.ClientSession that does nothing."""
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        return session

    def test_returns_string_in_normal_context(self):
        """In a plain synchronous context, asyncio.run() is called and returns its result."""
        # Patch dispatch layer: no running loop → asyncio.run() is the path taken
        with patch("asyncio.get_running_loop", side_effect=RuntimeError("no loop")), \
             patch("asyncio.run", return_value="mocked output") as mock_run:
            result = scrape_topic("AI engineering", n_google=0, n_reddit=0, n_github=0)
        mock_run.assert_called_once()
        self.assertIsInstance(result, str)

    def test_no_error_when_no_api_keys(self):
        """With no API keys or subreddits, scrape_topic() returns a string without error."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError("no loop")), \
             patch("asyncio.run", return_value=""):
            result = scrape_topic(
                "test topic",
                n_google=0,
                n_reddit=0,
                n_github=0,
                subreddits=[],
            )
        self.assertIsInstance(result, str)

    def test_uses_thread_pool_when_loop_is_running(self):
        """
        When asyncio.get_running_loop() does NOT raise, scrape_topic() must
        submit work to a ThreadPoolExecutor instead of calling asyncio.run()
        directly — which would raise RuntimeError('This event loop is already running').
        """
        # Simulate: asyncio.get_running_loop() succeeds (loop is running)
        fake_loop = MagicMock()

        future_result = "mocked result"

        class _FakeExecutor:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def submit(self, fn, *args, **kwargs):
                f = MagicMock()
                f.result.return_value = future_result
                return f

        with patch("asyncio.get_running_loop", return_value=fake_loop), \
             patch("concurrent.futures.ThreadPoolExecutor", return_value=_FakeExecutor()), \
             patch("asyncio.run", side_effect=AssertionError("asyncio.run must NOT be called when loop is running")):
            result = scrape_topic("test topic", n_google=0, n_reddit=0, n_github=0)

        self.assertEqual(result, future_result)

    def test_uses_asyncio_run_when_no_loop(self):
        """
        When asyncio.get_running_loop() raises RuntimeError (no running loop),
        scrape_topic() must call asyncio.run() directly.
        """
        async def _fake_run(coro):
            return "direct result"

        with patch("asyncio.get_running_loop", side_effect=RuntimeError("no loop")), \
             patch("asyncio.run", side_effect=lambda coro: "direct result") as mock_run, \
             patch("aiohttp.ClientSession", return_value=self._make_mock_session()):
            result = scrape_topic("test topic", n_google=0, n_reddit=0, n_github=0)

        mock_run.assert_called_once()
        self.assertEqual(result, "direct result")


if __name__ == "__main__":
    unittest.main(verbosity=2)
