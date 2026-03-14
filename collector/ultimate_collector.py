"""
UltimateCollector — unified async document collector for UltimateDocResearcher.

Sources supported:
  • Local PDFs / Markdown / text files
  • Web scraping (Google CSE, GitHub, Reddit, arbitrary URLs)
  • Google Drive (Docs, PDFs, Sheets → plain text)
  • GitHub repositories (README + code files)

Output: data/all_docs.txt  (one document per line, UTF-8)
         data/metadata.jsonl (title, source, url, chars per doc)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import aiohttp
import aiofiles

# ── Optional heavy deps (graceful degradation) ────────────────────────────────
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from googleapiclient.discovery import build as gdrive_build
    from google.oauth2.service_account import Credentials as SACredentials
    from google.oauth2.credentials import Credentials as OAuthCredentials
    HAS_GDRIVE = True
except ImportError:
    HAS_GDRIVE = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

logger = logging.getLogger("ultimate_collector")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class Document:
    title: str
    text: str
    source: str          # "pdf" | "web" | "drive" | "github" | "reddit" | "local"
    url: str = ""
    chars: int = 0

    def __post_init__(self):
        self.text = self._clean(self.text)
        self.chars = len(self.text)

    @staticmethod
    def _clean(text: str) -> str:
        """Normalize whitespace, drop zero-width characters."""
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @property
    def doc_id(self) -> str:
        return hashlib.sha1((self.url + self.title).encode()).hexdigest()[:12]


# ── Source path safety ────────────────────────────────────────────────────────

# Folder names commonly used for personal files — not research sources
_PERSONAL_FOLDER_NAMES = {
    "downloads", "documents", "desktop", "pictures", "photos",
    "videos", "music", "dropbox", "onedrive", "icloud drive",
    "google drive", "my documents", "my downloads",
}


def _warn_if_personal_folder(path: Path) -> None:
    """
    Print a clear warning when --pdf-dir points at a folder that typically
    contains personal files rather than research documents.

    This doesn't block execution — the user may intentionally have research
    PDFs there — but it surfaces the risk before the run starts.
    """
    folder_name = path.name.lower()
    parent_name = path.parent.name.lower()

    is_risky = (
        folder_name in _PERSONAL_FOLDER_NAMES
        or parent_name in _PERSONAL_FOLDER_NAMES
        # e.g. C:/Users/Dana/Downloads or ~/Downloads
        or any(part.lower() in _PERSONAL_FOLDER_NAMES for part in path.parts)
    )

    if is_risky:
        print(
            f"\n⚠️  WARNING: --pdf-dir points at '{path}'\n"
            f"   This looks like a personal folder, not a research-specific directory.\n"
            f"   It may contain invoices, contracts, photos, and other personal files\n"
            f"   that will contaminate your research corpus.\n"
            f"\n"
            f"   Recommended: create a dedicated papers/ folder and copy only your\n"
            f"   research PDFs there before running the collector:\n"
            f"\n"
            f"       mkdir papers\n"
            f"       cp ~/Downloads/my-research-paper.pdf papers/\n"
            f"       python -m collector.ultimate_collector --pdf-dir papers/ ...\n"
            f"\n"
            f"   The analyzer will still filter out personal documents, but a clean\n"
            f"   source folder is faster and avoids accidental data leakage.\n"
        )


# ── Sub-collectors ─────────────────────────────────────────────────────────────

class PDFCollector:
    """Extract text from local PDF / Markdown / text files."""

    def __init__(self, paths: List[str | Path]):
        self.paths = [Path(p) for p in paths]

    def collect(self) -> List[Document]:
        docs: List[Document] = []
        for path in self.paths:
            if not path.exists():
                logger.warning("File not found: %s", path)
                continue
            suffix = path.suffix.lower()
            try:
                if suffix == ".pdf":
                    docs.append(self._extract_pdf(path))
                elif suffix in (".md", ".txt", ".rst"):
                    docs.append(self._extract_text(path))
                else:
                    logger.debug("Skipping unsupported file type: %s", path)
            except Exception as exc:
                logger.error("Failed to read %s: %s", path, exc)
        return docs

    def _extract_pdf(self, path: Path) -> Document:
        if not HAS_PYMUPDF:
            raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")
        doc = fitz.open(str(path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return Document(
            title=path.stem,
            text="\n\n".join(pages),
            source="pdf",
            url=f"file://{path.resolve()}",
        )

    def _extract_text(self, path: Path) -> Document:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return Document(
            title=path.stem,
            text=text,
            source="local",
            url=f"file://{path.resolve()}",
        )


class WebScraper:
    """
    Scrape content from:
      - Arbitrary URLs
      - Google Custom Search API results
      - Reddit posts/comments (JSON API, no auth required)
      - GitHub repo README + source files
    """

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    REDDIT_HEADERS = {**HEADERS, "Accept": "application/json"}

    def __init__(
        self,
        urls: Optional[List[str]] = None,
        google_queries: Optional[List[str]] = None,
        reddit_subreddits: Optional[List[str]] = None,
        github_repos: Optional[List[str]] = None,  # ["owner/repo", ...]
        google_api_key: str = "",
        google_cx: str = "",
        max_results_per_query: int = 5,
        timeout: int = 20,
    ):
        self.urls = urls or []
        self.google_queries = google_queries or []
        self.reddit_subreddits = reddit_subreddits or []
        self.github_repos = github_repos or []
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY", "")
        self.google_cx = google_cx or os.getenv("GOOGLE_CX", "")
        self.max_results = max_results_per_query
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def collect_async(self) -> List[Document]:
        async with aiohttp.ClientSession(
            headers=self.HEADERS, timeout=self.timeout
        ) as session:
            tasks = []
            for url in self.urls:
                tasks.append(self._fetch_url(session, url))
            for query in self.google_queries:
                tasks.append(self._google_search(session, query))
            for sub in self.reddit_subreddits:
                tasks.append(self._reddit_scrape(session, sub))
            for repo in self.github_repos:
                tasks.append(self._github_repo(session, repo))

            results = await asyncio.gather(*tasks, return_exceptions=True)

        docs: List[Document] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Scrape task failed: %s", r)
            elif isinstance(r, list):
                docs.extend(r)
            elif isinstance(r, Document):
                docs.append(r)
        return docs

    def collect(self) -> List[Document]:
        return asyncio.run(self.collect_async())

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _fetch_url(self, session: aiohttp.ClientSession, url: str) -> Document:
        async with session.get(url) as resp:
            resp.raise_for_status()
            html = await resp.text(errors="ignore")
        text = self._html_to_text(html)
        title = self._extract_title(html) or urlparse(url).netloc
        return Document(title=title, text=text, source="web", url=url)

    async def _google_search(
        self, session: aiohttp.ClientSession, query: str
    ) -> List[Document]:
        if not self.google_api_key or not self.google_cx:
            logger.warning("Google CSE not configured — skipping query: %s", query)
            return []
        params = {
            "key": self.google_api_key,
            "cx": self.google_cx,
            "q": query,
            "num": self.max_results,
        }
        async with session.get(
            "https://www.googleapis.com/customsearch/v1", params=params
        ) as resp:
            data = await resp.json()

        docs: List[Document] = []
        for item in data.get("items", []):
            url = item.get("link", "")
            snippet = item.get("snippet", "")
            title = item.get("title", url)
            try:
                page_doc = await self._fetch_url(session, url)
                docs.append(page_doc)
            except Exception:
                # Fall back to just the snippet
                docs.append(Document(title=title, text=snippet, source="web", url=url))
        return docs

    async def _reddit_scrape(
        self, session: aiohttp.ClientSession, subreddit: str
    ) -> List[Document]:
        url = f"https://www.reddit.com/r/{subreddit}/top.json?limit=25&t=month"
        async with session.get(
            url, headers=self.REDDIT_HEADERS
        ) as resp:
            data = await resp.json()

        docs: List[Document] = []
        for post in data.get("data", {}).get("children", []):
            pd = post["data"]
            title = pd.get("title", "")
            selftext = pd.get("selftext", "")
            permalink = "https://www.reddit.com" + pd.get("permalink", "")
            if selftext.strip():
                docs.append(
                    Document(
                        title=title,
                        text=f"# {title}\n\n{selftext}",
                        source="reddit",
                        url=permalink,
                    )
                )
        return docs

    async def _github_repo(
        self, session: aiohttp.ClientSession, repo: str
    ) -> List[Document]:
        """Download README + up to 20 .py / .md files from a GitHub repo."""
        api_base = f"https://api.github.com/repos/{repo}"
        gh_token = os.getenv("GITHUB_TOKEN", "")
        headers = {"Accept": "application/vnd.github+json"}
        if gh_token:
            headers["Authorization"] = f"Bearer {gh_token}"

        docs: List[Document] = []

        # README
        readme_url = f"{api_base}/readme"
        async with session.get(readme_url, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                import base64
                content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
                docs.append(
                    Document(
                        title=f"{repo} README",
                        text=content,
                        source="github",
                        url=f"https://github.com/{repo}",
                    )
                )

        # Tree
        tree_url = f"{api_base}/git/trees/HEAD?recursive=1"
        async with session.get(tree_url, headers=headers) as resp:
            if resp.status != 200:
                return docs
            tree_data = await resp.json()

        files = [
            item["path"]
            for item in tree_data.get("tree", [])
            if item["type"] == "blob"
            and re.search(r"\.(py|md|txt|ipynb)$", item["path"])
            and not item["path"].startswith(".")
        ][:20]

        for filepath in files:
            raw_url = f"https://raw.githubusercontent.com/{repo}/HEAD/{filepath}"
            try:
                async with session.get(raw_url) as resp:
                    text = await resp.text(errors="ignore")
                docs.append(
                    Document(
                        title=f"{repo}/{filepath}",
                        text=text,
                        source="github",
                        url=raw_url,
                    )
                )
            except Exception as exc:
                logger.debug("Failed to fetch %s: %s", filepath, exc)

        return docs

    @staticmethod
    def _html_to_text(html: str) -> str:
        if not HAS_BS4:
            # Naive strip
            return re.sub(r"<[^>]+>", " ", html)
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        return soup.get_text(separator="\n")

    @staticmethod
    def _extract_title(html: str) -> str:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""


class DriveExtractor:
    """
    Pull documents from a Google Drive folder.
    Supports: Google Docs (export as plain text), PDFs, Markdown files.

    Auth options (env vars):
      GDRIVE_SA_KEY_PATH  → path to service-account JSON  (preferred in CI)
      GDRIVE_OAUTH_TOKEN  → path to oauth2 token.json     (local dev)
    """

    MIME_EXPORT = {
        "application/vnd.google-apps.document": "text/plain",
        "application/vnd.google-apps.spreadsheet": "text/csv",
    }
    MIME_DOWNLOAD = {
        "application/pdf",
        "text/plain",
        "text/markdown",
    }

    def __init__(self, folder_ids: List[str], max_files: int = 50):
        if not HAS_GDRIVE:
            raise RuntimeError(
                "google-api-python-client not installed. "
                "Run: pip install google-api-python-client google-auth"
            )
        self.folder_ids = folder_ids
        self.max_files = max_files
        self._service = self._build_service()

    def _build_service(self):
        sa_path = os.getenv("GDRIVE_SA_KEY_PATH")
        oauth_path = os.getenv("GDRIVE_OAUTH_TOKEN")
        scopes = ["https://www.googleapis.com/auth/drive.readonly"]

        if sa_path and Path(sa_path).exists():
            creds = SACredentials.from_service_account_file(sa_path, scopes=scopes)
        elif oauth_path and Path(oauth_path).exists():
            creds = OAuthCredentials.from_authorized_user_file(oauth_path, scopes)
        else:
            raise RuntimeError(
                "Set GDRIVE_SA_KEY_PATH or GDRIVE_OAUTH_TOKEN env var."
            )
        return gdrive_build("drive", "v3", credentials=creds, cache_discovery=False)

    def collect(self) -> List[Document]:
        docs: List[Document] = []
        for folder_id in self.folder_ids:
            docs.extend(self._list_folder(folder_id))
        return docs

    def _list_folder(self, folder_id: str) -> List[Document]:
        docs: List[Document] = []
        query = f"'{folder_id}' in parents and trashed=false"
        fields = "files(id,name,mimeType,webViewLink)"
        resp = (
            self._service.files()
            .list(q=query, fields=fields, pageSize=self.max_files)
            .execute()
        )
        for f in resp.get("files", []):
            try:
                doc = self._fetch_file(f)
                if doc:
                    docs.append(doc)
            except Exception as exc:
                logger.error("Drive file %s failed: %s", f["name"], exc)
        return docs

    def _fetch_file(self, f: dict) -> Optional[Document]:
        mime = f["mimeType"]
        file_id = f["id"]
        name = f["name"]
        url = f.get("webViewLink", f"https://drive.google.com/file/d/{file_id}")

        if mime in self.MIME_EXPORT:
            export_mime = self.MIME_EXPORT[mime]
            resp = (
                self._service.files()
                .export(fileId=file_id, mimeType=export_mime)
                .execute()
            )
            text = resp.decode("utf-8", errors="ignore") if isinstance(resp, bytes) else resp
            return Document(title=name, text=text, source="drive", url=url)

        if mime in self.MIME_DOWNLOAD:
            request = self._service.files().get_media(fileId=file_id)
            content = request.execute()
            if mime == "application/pdf":
                if not HAS_PYMUPDF:
                    logger.warning("PyMuPDF needed for Drive PDF: %s", name)
                    return None
                doc = fitz.open(stream=content, filetype="pdf")
                text = "\n\n".join(p.get_text() for p in doc)
                doc.close()
            else:
                text = content.decode("utf-8", errors="ignore")
            return Document(title=name, text=text, source="drive", url=url)

        logger.debug("Unsupported Drive mime %s for %s", mime, name)
        return None


# ── Main UltimateCollector ─────────────────────────────────────────────────────

class UltimateCollector:
    """
    Orchestrates all sub-collectors and writes unified output.

    Usage:
        collector = UltimateCollector(
            pdf_paths=["papers/"],
            google_queries=["Claude skills optimization site:arxiv.org"],
            reddit_subreddits=["MachineLearning", "LocalLLaMA"],
            github_repos=["karpathy/autoresearch"],
            drive_folder_ids=["1AbCdEf..."],
            output_dir="data/",
        )
        collector.run()
    """

    def __init__(
        self,
        pdf_paths: Optional[List[str]] = None,
        urls: Optional[List[str]] = None,
        google_queries: Optional[List[str]] = None,
        reddit_subreddits: Optional[List[str]] = None,
        github_repos: Optional[List[str]] = None,
        drive_folder_ids: Optional[List[str]] = None,
        google_api_key: str = "",
        google_cx: str = "",
        output_dir: str = "data/",
        min_chars: int = 200,
        dedup: bool = True,
    ):
        # Expand glob patterns in pdf_paths
        resolved_paths: List[Path] = []
        for p in pdf_paths or []:
            path = Path(p)
            if path.is_dir():
                _warn_if_personal_folder(path)
                resolved_paths.extend(path.rglob("*.pdf"))
                resolved_paths.extend(path.rglob("*.md"))
                resolved_paths.extend(path.rglob("*.txt"))
            else:
                resolved_paths.append(path)

        self.pdf_collector = PDFCollector(resolved_paths) if resolved_paths else None
        self.web_scraper = WebScraper(
            urls=urls,
            google_queries=google_queries,
            reddit_subreddits=reddit_subreddits,
            github_repos=github_repos,
            google_api_key=google_api_key,
            google_cx=google_cx,
        ) if (urls or google_queries or reddit_subreddits or github_repos) else None

        self.drive_extractor = None
        if drive_folder_ids and HAS_GDRIVE:
            try:
                self.drive_extractor = DriveExtractor(drive_folder_ids)
            except RuntimeError as exc:
                logger.warning("Drive disabled: %s", exc)

        self.output_dir = Path(output_dir)
        self.min_chars = min_chars
        self.dedup = dedup

    def run(self) -> List[Document]:
        logger.info("=== UltimateCollector starting ===")
        all_docs: List[Document] = []

        if self.pdf_collector:
            logger.info("Collecting PDFs / local files…")
            docs = self.pdf_collector.collect()
            logger.info("  → %d documents", len(docs))
            all_docs.extend(docs)

        if self.web_scraper:
            logger.info("Scraping web sources…")
            docs = self.web_scraper.collect()
            logger.info("  → %d documents", len(docs))
            all_docs.extend(docs)

        if self.drive_extractor:
            logger.info("Extracting Google Drive…")
            docs = self.drive_extractor.collect()
            logger.info("  → %d documents", len(docs))
            all_docs.extend(docs)

        # Filter short docs
        all_docs = [d for d in all_docs if d.chars >= self.min_chars]

        # Deduplicate by doc_id (hash of url+title)
        if self.dedup:
            seen: set[str] = set()
            unique: List[Document] = []
            for d in all_docs:
                if d.doc_id not in seen:
                    seen.add(d.doc_id)
                    unique.append(d)
            all_docs = unique

        logger.info("Total documents after filter+dedup: %d", len(all_docs))
        self._write_output(all_docs)
        return all_docs

    def _write_output(self, docs: List[Document]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # all_docs.txt — one doc per block, separated by <DOC_SEP>
        all_docs_path = self.output_dir / "all_docs.txt"
        with all_docs_path.open("w", encoding="utf-8") as f:
            for doc in docs:
                f.write(f"=== {doc.title} [{doc.source}] ===\n")
                f.write(doc.text)
                f.write("\n\n<DOC_SEP>\n\n")
        logger.info("Wrote %s (%.1f MB)", all_docs_path, all_docs_path.stat().st_size / 1e6)

        # metadata.jsonl
        meta_path = self.output_dir / "metadata.jsonl"
        with meta_path.open("w", encoding="utf-8") as f:
            for doc in docs:
                row = {k: v for k, v in asdict(doc).items() if k != "text"}
                f.write(json.dumps(row) + "\n")
        logger.info("Wrote %s", meta_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UltimateCollector CLI")
    parser.add_argument("--pdf-dir", help="Directory of PDFs / docs")
    parser.add_argument("--urls", nargs="*", help="Specific URLs to scrape")
    parser.add_argument("--queries", nargs="*", help="Google search queries")
    parser.add_argument("--reddit", nargs="*", help="Subreddits to scrape")
    parser.add_argument("--github", nargs="*", help="GitHub repos (owner/repo)")
    parser.add_argument("--drive-folders", nargs="*", help="Google Drive folder IDs")
    parser.add_argument("--output-dir", default="data/", help="Output directory")
    parser.add_argument("--min-chars", type=int, default=200)
    args = parser.parse_args()

    # Auto-create --pdf-dir and --output-dir if they don't exist
    if args.pdf_dir:
        pdf_dir_path = Path(args.pdf_dir)
        if not pdf_dir_path.exists():
            pdf_dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {pdf_dir_path} (was missing)")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    collector = UltimateCollector(
        pdf_paths=[args.pdf_dir] if args.pdf_dir else None,
        urls=args.urls,
        google_queries=args.queries,
        reddit_subreddits=args.reddit,
        github_repos=args.github,
        drive_folder_ids=args.drive_folders,
        output_dir=args.output_dir,
        min_chars=args.min_chars,
    )
    docs = collector.run()
    print(f"\n✅ Collected {len(docs)} documents → {args.output_dir}all_docs.txt")
