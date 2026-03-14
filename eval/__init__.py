"""eval — standardized 5-criteria evaluation framework for UltimateDocResearcher."""
# Lazy imports prevent RuntimeWarning when running `python -m eval.run_eval`.

__all__ = ["evaluate", "load_spec", "compute_weighted_score"]


def __getattr__(name: str):
    if name in ("evaluate", "load_spec", "compute_weighted_score"):
        from .run_eval import evaluate, load_spec, compute_weighted_score  # noqa: PLC0415
        globals()["evaluate"] = evaluate
        globals()["load_spec"] = load_spec
        globals()["compute_weighted_score"] = compute_weighted_score
        return globals()[name]
    raise AttributeError(f"module 'eval' has no attribute {name!r}")
