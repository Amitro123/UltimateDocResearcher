"""
drive_extractor.py — Standalone Google Drive extractor.

Can also mount Colab/Kaggle drives and walk their file trees.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("drive_extractor")


def _try_colab_mount(mount_point: str = "/content/drive") -> bool:
    try:
        from google.colab import drive  # type: ignore
        drive.mount(mount_point)
        return True
    except ImportError:
        return False
    except Exception as exc:
        logger.warning("Colab mount failed: %s", exc)
        return False


def _try_kaggle_mount() -> Optional[str]:
    """Return '/kaggle/input' if running on Kaggle."""
    p = Path("/kaggle/input")
    return str(p) if p.exists() else None


def collect_from_mounted_drive(
    root: Optional[str] = None,
    extensions: tuple[str, ...] = (".pdf", ".md", ".txt", ".py"),
    max_files: int = 100,
) -> List[dict]:
    """
    Walk a mounted filesystem (Colab Drive / Kaggle input) and
    return a list of {path, text} dicts.
    """
    # Auto-detect environment
    if root is None:
        if kaggle_root := _try_kaggle_mount():
            root = kaggle_root
            logger.info("Kaggle input detected: %s", root)
        elif _try_colab_mount():
            root = "/content/drive/MyDrive"
            logger.info("Colab Drive mounted at %s", root)
        else:
            raise RuntimeError("No mounted drive found. Pass root= explicitly.")

    results: List[dict] = []
    for path in Path(root).rglob("*"):
        if len(results) >= max_files:
            break
        if path.suffix.lower() not in extensions:
            continue
        try:
            if path.suffix.lower() == ".pdf":
                text = _read_pdf(path)
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
            results.append({"path": str(path), "text": text})
        except Exception as exc:
            logger.debug("Skip %s: %s", path, exc)

    logger.info("Collected %d files from %s", len(results), root)
    return results


def _read_pdf(path: Path) -> str:
    try:
        import fitz
        doc = fitz.open(str(path))
        text = "\n\n".join(p.get_text() for p in doc)
        doc.close()
        return text
    except ImportError:
        raise RuntimeError("PyMuPDF required: pip install pymupdf")


# ── Kaggle Dataset downloader ─────────────────────────────────────────────────

def download_kaggle_dataset(
    dataset: str,
    dest: str = "/kaggle/working",
    unzip: bool = True,
) -> str:
    """
    Download a Kaggle dataset using the Kaggle CLI.
    Requires KAGGLE_USERNAME and KAGGLE_KEY env vars.

    Args:
        dataset: "owner/dataset-name"
        dest:    destination directory
        unzip:   unzip after download

    Returns:
        Path to downloaded directory.
    """
    import subprocess
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", dest]
    if unzip:
        cmd.append("--unzip")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed: {result.stderr}")
    logger.info("Dataset %s downloaded to %s", dataset, dest)
    return dest
