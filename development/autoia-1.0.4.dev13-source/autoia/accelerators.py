"""Optional non-trusted accelerator helpers for AutoIA applications.

AutoIA's V1 machine-rigorous proof kernel remains the audited CPU binary64
backend.  This module exposes optional Torch devices for *proposal/assist*
computations that can be performed in bulk and then independently verified by
AutoIA on CPU.  No value produced here is a certificate.

Apple Metal is exposed through PyTorch's ``mps`` backend; NVIDIA devices use
``cuda``.  Torch is intentionally an optional dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence
import os
import platform
import numpy as np

from .exceptions import InvalidInputError, UnsupportedOperationError
from .intervals import Interval


@dataclass(frozen=True)
class AcceleratorInfo:
    name: str
    available: bool
    trusted_for_proof: bool
    default_dtype: str
    detail: str
    device_name: str | None = None
    compute_capability: str | None = None
    torch_version: str | None = None
    host_architecture: str | None = None


@dataclass(frozen=True)
class ProposalBatchResult:
    values: np.ndarray
    device: str
    dtype: str
    machine_rigorous: bool = False


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def available_accelerators() -> dict[str, AcceleratorInfo]:
    """Report the trusted CPU proof backend and optional Torch proposal devices.

    The returned hardware metadata is diagnostic only.  In V1, neither MPS nor
    CUDA is part of the trusted proof base; accelerator results must be
    independently verified by AutoIA's CPU binary64 backend.
    """
    torch = _torch()
    host_arch = platform.machine() or None
    torch_version = getattr(torch, "__version__", None) if torch is not None else None
    out = {
        "cpu": AcceleratorInfo(
            "cpu", True, True, "float64",
            "Audited AutoIA binary64 proof backend; multiprocessing is supported for many boxes.",
            device_name=platform.processor() or "host CPU",
            torch_version=torch_version,
            host_architecture=host_arch,
        )
    }
    if torch is None:
        out["mps"] = AcceleratorInfo(
            "mps", False, False, "float32",
            "PyTorch is not installed; Apple Metal proposal acceleration is unavailable.",
            device_name="Apple Metal via PyTorch MPS", host_architecture=host_arch,
        )
        out["cuda"] = AcceleratorInfo(
            "cuda", False, False, "float64",
            "PyTorch is not installed; NVIDIA CUDA proposal acceleration is unavailable.",
            device_name="NVIDIA CUDA", host_architecture=host_arch,
        )
        return out

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    out["mps"] = AcceleratorInfo(
        "mps", mps_available, False, "float32",
        (
            "Apple Metal / PyTorch MPS proposal backend. Results are not proof values; "
            "AutoIA CPU verification remains required."
            if mps_available else
            "PyTorch MPS is not available on this machine/build."
        ),
        device_name="Apple Metal via PyTorch MPS",
        torch_version=torch_version, host_architecture=host_arch,
    )
    cuda_available = bool(getattr(torch, "cuda", None) is not None and torch.cuda.is_available())
    cuda_name = "NVIDIA CUDA"
    cuda_capability = None
    if cuda_available:
        try:
            cuda_name = str(torch.cuda.get_device_name(0))
        except Exception:
            pass
        try:
            major, minor = torch.cuda.get_device_capability(0)
            cuda_capability = f"{int(major)}.{int(minor)}"
        except Exception:
            pass
    out["cuda"] = AcceleratorInfo(
        "cuda", cuda_available, False, "float64",
        (
            "NVIDIA CUDA proposal backend. Results are not proof values; AutoIA CPU verification remains required."
            if cuda_available else
            "PyTorch CUDA is not available on this machine/build."
        ),
        device_name=cuda_name, compute_capability=cuda_capability,
        torch_version=torch_version, host_architecture=host_arch,
    )
    return out


def resolve_proposal_device(device: str = "auto") -> str:
    device = str(device).lower()
    info = available_accelerators()
    if device == "auto":
        # V1 conservative policy: proposal-device performance depends on the
        # arithmetic intensity of the caller's Torch function, not merely the
        # number of points.  AutoIA therefore avoids silently moving work to a
        # GPU.  Users/applications opt in with device="mps" or device="cuda".
        return "cpu"
    if device not in {"cpu", "mps", "cuda"}:
        raise InvalidInputError("proposal device must be 'auto', 'cpu', 'mps', or 'cuda'.")
    if not info[device].available:
        raise UnsupportedOperationError(info[device].detail)
    return device


def _normalize_points(points) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape((1, -1))
    if arr.ndim != 2 or not np.all(np.isfinite(arr)):
        raise InvalidInputError("proposal points must be a finite (n_points, dimension) array.")
    return arr


def proposal_map(function: Callable[[Any], Any], points, *, device: str = "auto",
                 dtype: str | None = None) -> ProposalBatchResult:
    """Evaluate a Torch-compatible batched proposal function on CPU/MPS/CUDA.

    ``device="auto"`` deliberately selects CPU in V1.  GPU proposal use is
    explicit because performance is workload-dependent.

    ``function`` receives one Torch tensor of shape ``(n_points, dimension)`` and
    should return a tensor-like batch with first dimension ``n_points``.  The
    result is explicitly non-rigorous and is suitable only for proposal,
    ranking, initialization, or other work that will later be CPU-certified.
    """
    torch = _torch()
    if torch is None:
        raise UnsupportedOperationError(
            "proposal_map requires optional PyTorch; install a PyTorch build appropriate for your machine."
        )
    dev = resolve_proposal_device(device)
    arr = _normalize_points(points)
    if dtype is None:
        dtype = "float32" if dev == "mps" else "float64"
    dtype = str(dtype).lower()
    allowed = {"float32": torch.float32, "float64": torch.float64}
    if dtype not in allowed:
        raise InvalidInputError("proposal dtype must be 'float32' or 'float64'.")
    if dev == "mps" and dtype == "float64":
        raise UnsupportedOperationError(
            "AutoIA's MPS proposal backend uses float32; request dtype='float32'. "
            "The rigorous proof remains CPU binary64 regardless."
        )
    tensor = torch.as_tensor(arr, dtype=allowed[dtype], device=torch.device(dev))
    with torch.no_grad():
        values = function(tensor)
    if not torch.is_tensor(values):
        values = torch.as_tensor(values, device=tensor.device)
    values = values.detach().to("cpu")
    return ProposalBatchResult(
        values=np.asarray(values.numpy()).copy(),
        device=dev,
        dtype=dtype,
        machine_rigorous=False,
    )


def box_centers(boxes: Sequence[Sequence[Interval | Sequence[float]]]) -> np.ndarray:
    centers = []
    dim = None
    for box in boxes:
        row = []
        for item in box:
            if isinstance(item, Interval):
                lo, hi = float(item.lo), float(item.hi)
            else:
                pair = list(item)
                if len(pair) != 2:
                    raise InvalidInputError("Each proposal box entry must be an Interval or length-2 pair.")
                lo, hi = float(pair[0]), float(pair[1])
            if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
                raise InvalidInputError("Proposal box endpoints must be finite and ordered.")
            row.append(0.5 * lo + 0.5 * hi)
        if dim is None:
            dim = len(row)
        elif len(row) != dim:
            raise InvalidInputError("All proposal boxes must have the same dimension.")
        centers.append(row)
    return np.asarray(centers, dtype=float)


def proposal_box_centers(function: Callable[[Any], Any], boxes, *, device: str = "auto",
                         dtype: str | None = None) -> ProposalBatchResult:
    """Evaluate a Torch-compatible proposal function at many box centers."""
    return proposal_map(function, box_centers(list(boxes)), device=device, dtype=dtype)


__all__ = [
    "AcceleratorInfo", "ProposalBatchResult", "available_accelerators",
    "resolve_proposal_device", "proposal_map", "box_centers", "proposal_box_centers",
]
