from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import import_module
from importlib.util import find_spec
from typing import Literal

import numpy as np

# "cuda"/"cpu" label the torch device of BoTorch results; the array backends
# used by tamubo.exactbo are "numpy" and "cupy".
BackendName = Literal["auto", "numpy", "cupy", "cuda", "cpu"]
SelectedBackend = Literal["numpy", "cupy", "cuda", "cpu"]
__all__ = [
    "BackendName",
    "SelectedBackend",
    "BackendInfo",
    "has_cupy",
    "resolve_backend",
    "get_array_module",
    "to_numpy",
]

@dataclass(frozen=True)
class BackendInfo:
    """Resolved backend configuration."""
    requested: BackendName
    selected: SelectedBackend
    cupy_available: bool

@cache
def has_cupy() -> bool:
    """Return True when cupy is installed and can see at least one GPU."""
    if find_spec("cupy") is None:
        return False
    import cupy

    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except (RuntimeError, OSError):
        # No driver/GPU on this node (e.g. a login node) or CUDA libraries missing.
        return False

@cache
def resolve_backend(backend: BackendName = "auto") -> BackendInfo:
    """
    Resolve execution backend.

    Parameters
    ----------
    backend : {"auto", "numpy", "cupy", "cuda", "cpu"}, default="auto"
        Requested backend. ``"auto"`` picks cupy when a GPU is available,
        otherwise numpy.

    Returns
    -------
    BackendInfo
        Final backend selection plus availability information.
    """
    if backend not in ("auto", "numpy", "cupy", "cuda", "cpu"):
        raise ValueError(
            f"Unsupported backend '{backend}'. Choose from 'auto', 'numpy', 'cupy', 'cuda', 'cpu'."
        )
    cupy_available = has_cupy()
    if backend == "cupy" and not cupy_available:
        raise RuntimeError(
            "backend='cupy' was requested, but cupy is not installed or no GPU is visible. "
            "Install cupy-cuda12x and run on a GPU node, or use backend='numpy'."
        )
    if backend == "auto":
        selected: SelectedBackend = "cupy" if cupy_available else "numpy"
    else:
        selected = backend
    return BackendInfo(requested=backend, selected=selected, cupy_available=cupy_available)

def get_array_module(backend: BackendName = "auto"):
    """Return the array module (`numpy` or `cupy`) for a backend."""
    selected = resolve_backend(backend).selected
    if selected == "numpy":
        return np
    if selected == "cupy":
        return import_module("cupy")
    raise ValueError(
        f"backend='{selected}' is a torch device label, not an array backend; use 'numpy' or 'cupy'."
    )

def to_numpy(array):
    """Copy a numpy/cupy array (or scalar) to a host numpy array."""
    # cupy forbids implicit np.asarray() on device arrays; .get() is its explicit copy.
    if type(array).__module__.split(".")[0] == "cupy":
        return array.get()
    return np.asarray(array)
