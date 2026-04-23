from __future__ import annotations

import cupynumeric as cp

from tamubo.utils import BackendInfo, BackendName, resolve_backend

__all__ = ["cp", "require_cupynumeric_backend"]


def require_cupynumeric_backend(backend: BackendName = "auto") -> BackendInfo:
    """
    Resolve and validate the ExactBO backend.

    ExactBO now runs only on the cuPyNumeric backend. ``backend="auto"`` is still
    accepted for compatibility, but it must resolve to ``"cupynumeric"``.
    """
    backend_info = resolve_backend(backend)
    if backend_info.selected == "cupynumeric":
        return backend_info

    if backend_info.requested == "auto":
        raise ImportError(
            "ExactBO requires cuPyNumeric, but backend='auto' resolved to 'numpy'. "
            "Install cuPyNumeric or request backend='cupynumeric' explicitly."
        )

    raise ValueError(
        "ExactBO supports only backend='cupynumeric' (or backend='auto' when "
        "cuPyNumeric is available)."
    )
