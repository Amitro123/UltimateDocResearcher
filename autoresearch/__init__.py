# Lazy imports prevent RuntimeWarning when running submodules via `python -m`.
# e.g. `python -m autoresearch.train` no longer warns about sys.modules collision.

__all__ = ["prepare", "train", "TrainConfig", "research_loop"]


def __getattr__(name: str):
    if name == "prepare":
        from .prepare import prepare  # noqa: PLC0415
        globals()["prepare"] = prepare
        return prepare
    if name in ("train", "TrainConfig", "research_loop"):
        from .train import train, TrainConfig, research_loop  # noqa: PLC0415
        globals()["train"] = train
        globals()["TrainConfig"] = TrainConfig
        globals()["research_loop"] = research_loop
        return globals()[name]
    raise AttributeError(f"module 'autoresearch' has no attribute {name!r}")
