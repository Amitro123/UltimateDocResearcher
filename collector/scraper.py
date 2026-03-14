"""
scraper.py — Specialized scrapers extracted from UltimateCollector.

Exposes standalone async helpers and a synchronous `scrape_topic()` function
that's convenient to call from Jupyter / Kaggle notebooks.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import List, Optional
from urllib.parse import urlencode

import aiohttp

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


# ── Public helpers ─────────────────────────────────────────────────────────────

async def scrape_url(url: str, session: aiohttp.ClientSession) -> str:
    """Return clean text from a single URL."""
    headers = {"User-Agent": "Mozilla/5.0 UltimateDocResearcher/1.0"}
    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
        resp.raise_for_status()
        html = await resp.text(errors="ignore")
    return _html_to_text(html)


async def google_search_urls(
    query: str,
    api_key: str,
    cx: str,
    n: int = 5,
    session: Optional[aiohttp.ClientSession] = None,
) -> List[str]:
    """Return list of URLs from Google Custom Search."""
    params = {"key": api_key, "cx": cx, "q": query, "num": n}
    url = "https://www.googleapis.com/customsearch/v1?" + urlencode(params)
    close = session is None
    if session is None:
        session = aiohttp.ClientSession()
    try:
        async with session.get(url) as resp:
            data = await resp.json()
        return [item["link"] for item in data.get("items", [])]
    finally:
        if close:
            await session.close()


async def reddit_top_posts(
    subreddit: str,
    n: int = 25,
    timeframe: str = "month",
    session: Optional[aiohttp.ClientSession] = None,
) -> List[dict]:
    """Return top posts from a subreddit as dicts."""
    headers = {
        "User-Agent": "Mozilla/5.0 UltimateDocResearcher/1.0",
        "Accept": "application/json",
    }
    url = f"https://www.reddit.com/r/{subreddit}/top.json?limit={n}&t={timeframe}"
    close = session is None
    if session is None:
        session = aiohttp.ClientSession()
    try:
        async with session.get(url, headers=headers) as resp:
            data = await resp.json()
        posts = []
        for child in data.get("data", {}).get("children", []):
            pd = child["data"]
            posts.append({
                "title": pd.get("title", ""),
                "text": pd.get("selftext", ""),
                "url": "https://www.reddit.com" + pd.get("permalink", ""),
                "score": pd.get("score", 0),
            })
        return posts
    finally:
        if close:
            await session.close()


async def github_search_repos(
    query: str,
    n: int = 5,
    session: Optional[aiohttp.ClientSession] = None,
) -> List[str]:
    """Search GitHub and return list of 'owner/repo' strings."""
    token = os.getenv("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    params = {"q": query, "sort": "stars", "per_page": n}
    url = "https://api.github.com/search/repositories?" + urlencode(params)
    close = session is None
    if session is None:
        session = aiohttp.ClientSession()
    try:
        async with session.get(url, headers=headers) as resp:
            data = await resp.json()
        return [item["full_name"] for item in data.get("items", [])]
    finally:
        if close:
            await session.close()


# ── Convenience synchronous wrapper ──────────────────────────────────────────

def scrape_topic(
    topic: str,
    n_google: int = 5,
    n_reddit: int = 10,
    n_github: int = 3,
    subreddits: Optional[List[str]] = None,
    google_api_key: str = "",
    google_cx: str = "",
) -> str:
    """
    One-shot blocking call: collect web + reddit + github text for a topic.
    Returns all text concatenated (ready for prepare.py).
    """
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY", "")
    cx = google_cx or os.getenv("GOOGLE_CX", "")
    subs = subreddits or ["MachineLearning", "LocalLLaMA", "artificial"]

    async def _run() -> str:
        parts: List[str] = []
        async with aiohttp.ClientSession() as sess:
            # Google
            if api_key and cx:
                urls = await google_search_urls(topic, api_key, cx, n_google, sess)
                for url in urls:
                    try:
                        text = await scrape_url(url, sess)
                        parts.append(f"=== WEB: {url} ===\n{text[:8000]}")
                    except Exception:
                        pass

            # Reddit
            for sub in subs:
                posts = await reddit_top_posts(sub, n_reddit, session=sess)
                for p in posts:
                    if topic.lower() in (p["title"] + p["text"]).lower():
                        parts.append(f"=== REDDIT r/{sub}: {p['title']} ===\n{p['text'][:4000]}")

            # GitHub
            repos = await github_search_repos(topic, n_github, sess)
            # (actual file fetching handled by WebScraper._github_repo in the collector)
            for repo in repos:
                parts.append(f"=== GITHUB: {repo} ===\nRepo: https://github.com/{repo}\n")

        return "\n\n<DOC_SEP>\n\n".join(parts)

    return asyncio.run(_run())


# ── Utilities ─────────────────────────────────────────────────────────────────

def _html_to_text(html: str) -> str:
    if HAS_BS4:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n")
    return re.sub(r"<[^>]+>", " ", html)
