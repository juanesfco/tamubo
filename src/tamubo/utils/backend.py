from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Literal

BackendName = Literal["auto", "numpy", "cupynumeric", "cuda", "cpu"]
SelectedBackend = Literal["numpy", "cupynumeric", "cuda", "cpu"]
__all__ = ["BackendName", "SelectedBackend", "BackendInfo", "has_cupynumeric", "resolve_backend"]

@dataclass(frozen=True)
class BackendInfo:
    """Resolved backend configuration."""
    requested: BackendName
    selected: SelectedBackend
    cupynumeric_available: bool

def has_cupynumeric() -> bool:
    """Return True when cupynumeric can be imported in this environment."""
    return find_spec("cupynumeric") is not None

def resolve_backend(backend: BackendName = "auto") -> BackendInfo:
    """
    Resolve execution backend.

    Parameters
    ----------
    backend : {"auto", "numpy", "cupynumeric", "cuda", "cpu"}, default="auto"
        Requested backend.

    Returns
    -------
    BackendInfo
        Final backend selection plus availability information.
    """
    cupynumeric_available = has_cupynumeric()
    if backend not in ("auto", "numpy", "cupynumeric", "cuda", "cpu"):
        raise ValueError(
            f"Unsupported backend '{backend}'. Choose from 'auto', 'numpy', 'cupynumeric', 'cuda', 'cpu'."
        )
    if backend == "numpy":
        return BackendInfo(
            requested="numpy",
            selected="numpy",
            cupynumeric_available=cupynumeric_available,
        )
    if backend == "cupynumeric":
        if not cupynumeric_available:
            raise ImportError(
                "backend='cupynumeric' was requested, but cupynumeric is not installed. "
                "Install cupynumeric/legate or use backend='numpy' or backend='auto'."
            )
        return BackendInfo(
            requested="cupynumeric",
            selected="cupynumeric",
            cupynumeric_available=True,
        )
    if backend == "cuda":
        return BackendInfo(
            requested="cuda",
            selected="cuda",
            cupynumeric_available=cupynumeric_available,
        )
    if backend == "cpu":
        return BackendInfo(
            requested="cpu",
            selected="cpu",
            cupynumeric_available=cupynumeric_available,
        )
    selected: SelectedBackend = "cupynumeric" if cupynumeric_available else "numpy"
    return BackendInfo(
        requested="auto",
        selected=selected,
        cupynumeric_available=cupynumeric_available,
    )
