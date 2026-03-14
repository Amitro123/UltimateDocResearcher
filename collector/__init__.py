# Lazy imports prevent RuntimeWarning when running submodules via `python -m`.
# e.g. `python -m collector.ultimate_collector` no longer warns that
# 'collector.ultimate_collector' was found in sys.modules before execution.

__all__ = ["UltimateCollector", "Document", "scrape_topic", "analyze_corpus"]


def __getattr__(name: str):
    if name in ("UltimateCollector", "Document"):
        from .ultimate_collector import UltimateCollector, Document  # noqa: PLC0415
        globals()["UltimateCollector"] = UltimateCollector
        globals()["Document"] = Document
        return globals()[name]
    if name == "scrape_topic":
        from .scraper import scrape_topic  # noqa: PLC0415
        globals()["scrape_topic"] = scrape_topic
        return scrape_topic
    if name == "analyze_corpus":
        from .analyzer import analyze_corpus  # noqa: PLC0415
        globals()["analyze_corpus"] = analyze_corpus
        return analyze_corpus
    raise AttributeError(f"module 'collector' has no attribute {name!r}")
