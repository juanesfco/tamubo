"""Narrow transparent semantics for supported fitted exact Gaussian processes.

This module does *not* implement arbitrary Torch execution.  It recognizes a
small, auditable fitted-state subset and reconstructs posterior mean/variance
using AutoIA scalar arithmetic when an uncertain query reaches a fitted model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
import numpy as np


_SUPPORTED_KERNELS = {"RBFKernel", "MaternKernel"}
_SUPPORTED_MEANS = {"ZeroMean", "ConstantMean"}


def _class_name(obj: Any) -> str:
    return type(obj).__name__


def _class_lineage(cls: type) -> tuple[type, ...]:
    """Return one class MRO without importing optional GP packages."""
    return tuple(getattr(cls, "__mro__", (cls,)))


def _has_gpytorch_exactgp_lineage(cls: type) -> bool:
    return any(
        str(getattr(base, "__module__", "")) == "gpytorch.models.exact_gp"
        and str(getattr(base, "__name__", "")) == "ExactGP"
        for base in _class_lineage(cls)
    )


def _has_botorch_single_task_lineage(cls: type) -> bool:
    return any(
        str(getattr(base, "__module__", "")).startswith("botorch.models")
        and str(getattr(base, "__name__", "")) == "SingleTaskGP"
        for base in _class_lineage(cls)
    )


def _has_sklearn_gpr_lineage(cls: type) -> bool:
    """Recognize sklearn GaussianProcessRegressor without importing sklearn."""
    return any(
        str(getattr(base, "__module__", "")).startswith("sklearn.gaussian_process")
        and str(getattr(base, "__name__", "")) == "GaussianProcessRegressor"
        for base in _class_lineage(cls)
    )


def _as_numpy(value: Any, *, dtype=float) -> np.ndarray:
    """Copy tensor/array-like stored numeric state to CPU NumPy without mutation."""
    x = value
    detach = getattr(x, "detach", None)
    if callable(detach):
        x = detach()
    cpu = getattr(x, "cpu", None)
    if callable(cpu):
        x = cpu()
    numpy = getattr(x, "numpy", None)
    if callable(numpy):
        x = numpy()
    return np.asarray(x, dtype=dtype).copy()


def _scalar(value: Any) -> float:
    a = _as_numpy(value).reshape(-1)
    if a.size != 1:
        raise NotImplementedError("Expected one stored scalar GP parameter.")
    out = float(a[0])
    if not np.isfinite(out):
        raise ValueError("GP state contains a nonfinite stored parameter.")
    return out


def _vector(value: Any, d: int) -> np.ndarray:
    a = _as_numpy(value).reshape(-1)
    if a.size == 1:
        a = np.repeat(a, d)
    if a.size != d:
        raise NotImplementedError(
            f"Expected scalar or {d}-component GP parameter; received {a.size}."
        )
    if np.any(~np.isfinite(a)):
        raise ValueError("GP state contains nonfinite stored parameters.")
    return np.asarray(a, dtype=float)


def _unwrap_kernel(module: Any):
    outputscale = 1.0
    current = module
    if _class_name(current) == "ScaleKernel":
        outputscale = _scalar(getattr(current, "outputscale"))
        current = getattr(current, "base_kernel")
    name = _class_name(current)
    if name not in _SUPPORTED_KERNELS:
        raise NotImplementedError(
            "AutoIA transparent GP semantics currently support only RBFKernel "
            "and MaternKernel, optionally wrapped by ScaleKernel; received " + name + "."
        )
    return current, name, outputscale


def _unwrap_sklearn_kernel(module: Any):
    """Map sklearn RBF/Matérn kernels onto the common exact-GP kernel state.

    Supported forms are RBF or Matern, optionally multiplied by exactly one
    ConstantKernel on either side.  WhiteKernel and additive/composite kernels
    deliberately fail closed in this first sklearn coverage pass.
    """
    outputscale = 1.0
    current = module
    name = _class_name(current)
    if name == "Product":
        k1 = getattr(current, "k1", None)
        k2 = getattr(current, "k2", None)
        n1, n2 = _class_name(k1), _class_name(k2)
        if n1 == "ConstantKernel" and n2 in {"RBF", "Matern"}:
            outputscale = _scalar(getattr(k1, "constant_value"))
            current = k2
        elif n2 == "ConstantKernel" and n1 in {"RBF", "Matern"}:
            outputscale = _scalar(getattr(k2, "constant_value"))
            current = k1
        else:
            raise NotImplementedError(
                "AutoIA sklearn GP semantics currently support only RBF or Matern, "
                "optionally multiplied by one ConstantKernel."
            )
    name = _class_name(current)
    if name == "RBF":
        kind = "RBFKernel"
    elif name == "Matern":
        kind = "MaternKernel"
    else:
        raise NotImplementedError(
            "AutoIA sklearn GP semantics currently support only RBF or Matern, "
            "optionally multiplied by one ConstantKernel; received " + name + "."
        )
    if outputscale <= 0.0:
        raise ValueError("Kernel output scale must be positive.")
    return current, kind, float(outputscale)


def _extract_affine_input_transform(transform: Any, d: int):
    if transform is None:
        return np.zeros(d), np.ones(d)
    name = _class_name(transform)
    if name == "Normalize":
        offset = getattr(transform, "offset", None)
        coefficient = getattr(transform, "coefficient", None)
        if offset is None or coefficient is None:
            raise NotImplementedError("Normalize transform does not expose offset/coefficient.")
        return _vector(offset, d), _vector(coefficient, d)
    if name == "InputStandardize":
        means = getattr(transform, "means", None)
        stdvs = getattr(transform, "stdvs", None)
        if means is None or stdvs is None:
            raise NotImplementedError("InputStandardize does not expose means/stdvs.")
        return _vector(means, d), _vector(stdvs, d)
    raise NotImplementedError(
        "AutoIA transparent GP semantics support only no input transform, "
        "Normalize, or InputStandardize; received " + name + "."
    )


def _extract_outcome_transform(transform: Any):
    if transform is None:
        return 0.0, 1.0
    name = _class_name(transform)
    if name != "Standardize":
        raise NotImplementedError(
            "AutoIA transparent GP semantics support only no outcome transform "
            "or scalar Standardize; received " + name + "."
        )
    means = getattr(transform, "means", None)
    stdvs = getattr(transform, "stdvs", None)
    if means is None or stdvs is None:
        raise NotImplementedError("Standardize does not expose means/stdvs.")
    return _scalar(means), _scalar(stdvs)


def _kernel_numeric(kind: str, nu: float | None, outputscale: float,
                    lengthscale: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    z = (X[:, None, :] - Y[None, :, :]) / lengthscale.reshape(1, 1, -1)
    r2 = np.sum(z * z, axis=-1)
    if kind == "RBFKernel":
        base = np.exp(-0.5 * r2)
    else:
        r = np.sqrt(np.maximum(r2, 0.0))
        if nu == 0.5:
            base = np.exp(-r)
        elif nu == 1.5:
            c = math.sqrt(3.0)
            base = (1.0 + c * r) * np.exp(-c * r)
        elif nu == 2.5:
            c = math.sqrt(5.0)
            base = (1.0 + c * r + (5.0 / 3.0) * r2) * np.exp(-c * r)
        else:  # pragma: no cover - freeze rejects first
            raise NotImplementedError("Unsupported Matern nu.")
    return outputscale * base


def _sqrt_value(x):
    if hasattr(x, "unary"):
        return x.unary("sqrt")
    return math.sqrt(float(x))


def _exp_value(x):
    if hasattr(x, "unary"):
        return x.unary("exp")
    return math.exp(float(x))


def _kernel_validated(kind: str, nu: float | None, outputscale: float,
                      lengthscale: np.ndarray, x, Xtrain: np.ndarray):
    xv = np.asarray(x, dtype=object).reshape(-1)
    d = Xtrain.shape[1]
    if xv.size != d:
        raise ValueError(f"GP query has dimension {xv.size}; fitted GP expects {d}.")
    out = []
    for row in Xtrain:
        r2 = 0.0
        for j in range(d):
            z = (xv[j] - float(row[j])) / float(lengthscale[j])
            r2 = r2 + z ** 2
        if kind == "RBFKernel":
            k = _exp_value(-0.5 * r2)
        else:
            # Collapse only the nonnegative distance construction to interval form
            # if necessary. This is rigorous and avoids relying on a polynomial
            # lower bound remaining nonnegative at r=0.
            try:
                r = _sqrt_value(r2)
            except Exception:
                to_interval = getattr(r2, "to_interval", None)
                if not callable(to_interval):
                    raise
                from ..enclosures import ValidatedValue
                r = ValidatedValue(to_interval(), getattr(r2, "natural_interval", None)).unary("sqrt")
            if nu == 0.5:
                k = _exp_value(-r)
            elif nu == 1.5:
                c = math.sqrt(3.0)
                k = (1.0 + c * r) * _exp_value(-c * r)
            elif nu == 2.5:
                c = math.sqrt(5.0)
                k = (1.0 + c * r + (5.0 / 3.0) * r2) * _exp_value(-c * r)
            else:  # pragma: no cover
                raise NotImplementedError("Unsupported Matern nu.")
        out.append(float(outputscale) * k)
    return np.asarray(out, dtype=object)


def _kernel_pair_validated(kind: str, nu: float | None, outputscale: float,
                           lengthscale: np.ndarray, x, y):
    """Validated stationary kernel between two possibly uncertain query points."""
    xv = np.asarray(x, dtype=object).reshape(-1)
    yv = np.asarray(y, dtype=object).reshape(-1)
    if xv.size != yv.size or xv.size != len(lengthscale):
        raise ValueError("GP query-pair dimensions do not match the fitted kernel.")
    r2 = 0.0
    for j in range(xv.size):
        z = (xv[j] - yv[j]) / float(lengthscale[j])
        r2 = r2 + z ** 2
    if kind == "RBFKernel":
        base = _exp_value(-0.5 * r2)
    else:
        try:
            r = _sqrt_value(r2)
        except Exception:
            to_interval = getattr(r2, "to_interval", None)
            if not callable(to_interval):
                raise
            from ..enclosures import ValidatedValue
            r = ValidatedValue(
                to_interval(), getattr(r2, "natural_interval", None)
            ).unary("sqrt")
        if nu == 0.5:
            base = _exp_value(-r)
        elif nu == 1.5:
            c = math.sqrt(3.0)
            base = (1.0 + c * r) * _exp_value(-c * r)
        elif nu == 2.5:
            c = math.sqrt(5.0)
            base = (1.0 + c * r + (5.0 / 3.0) * r2) * _exp_value(-c * r)
        else:  # pragma: no cover
            raise NotImplementedError("Unsupported Matern nu.")
    return float(outputscale) * base


@dataclass(frozen=True)
class FrozenExactGPState:
    train_x: np.ndarray
    alpha: np.ndarray
    chol: np.ndarray
    mean_constant: float
    kernel_kind: str
    matern_nu: float | None
    lengthscale: np.ndarray
    outputscale: float
    input_offset: np.ndarray
    input_scale: np.ndarray
    outcome_offset: float = 0.0
    outcome_scale: float = 1.0
    source_kind: str = "gpytorch"

    @property
    def dimension(self) -> int:
        return int(self.train_x.shape[1])


@dataclass(frozen=True)
class FrozenIndependentMultiOutputGPState:
    """Frozen state for independent outputs represented as BoTorch output batches.

    Each output is one ordinary ``FrozenExactGPState``.  This intentionally does
    not represent correlated multi-task GPs; it only captures the standard
    multi-output ``SingleTaskGP`` construction in which outputs are independent.
    """
    outputs: tuple[FrozenExactGPState, ...]
    source_kind: str = "botorch-independent-multioutput"

    @property
    def dimension(self) -> int:
        if not self.outputs:
            raise ValueError("Independent multi-output GP state has no outputs.")
        return self.outputs[0].dimension

    @property
    def num_outputs(self) -> int:
        return len(self.outputs)


def _batched_scalars(value: Any, m: int, *, name: str) -> np.ndarray:
    a = _as_numpy(value).reshape(-1)
    if a.size == 1:
        a = np.repeat(a, m)
    if a.size != m:
        raise NotImplementedError(
            f"Expected one {name} per output ({m} values); received {a.size}."
        )
    if np.any(~np.isfinite(a)):
        raise ValueError(f"GP state contains nonfinite stored {name} values.")
    return np.asarray(a, dtype=float)


def _batched_vectors(value: Any, m: int, d: int, *, name: str) -> np.ndarray:
    a = _as_numpy(value)
    if a.size == 1:
        out = np.full((m, d), float(a.reshape(-1)[0]), dtype=float)
    elif a.size == d:
        out = np.repeat(a.reshape(1, d), m, axis=0)
    elif a.size == m * d:
        out = a.reshape(m, d)
    else:
        raise NotImplementedError(
            f"Expected shared or per-output {name} with {d} component(s); "
            f"received stored shape {a.shape}."
        )
    if np.any(~np.isfinite(out)):
        raise ValueError(f"GP state contains nonfinite stored {name} values.")
    return np.asarray(out, dtype=float)


def _extract_batched_kernel(module: Any, m: int, d: int):
    outputscale = np.ones(m, dtype=float)
    current = module
    if _class_name(current) == "ScaleKernel":
        outputscale = _batched_scalars(
            getattr(current, "outputscale"), m, name="kernel outputscale"
        )
        current = getattr(current, "base_kernel")
    name = _class_name(current)
    if name not in _SUPPORTED_KERNELS:
        raise NotImplementedError(
            "AutoIA transparent GP semantics currently support only RBFKernel "
            "and MaternKernel, optionally wrapped by ScaleKernel; received " + name + "."
        )
    lengthscales = _batched_vectors(
        getattr(current, "lengthscale"), m, d, name="kernel lengthscale"
    )
    if np.any(lengthscales <= 0.0) or np.any(outputscale <= 0.0):
        raise ValueError("Kernel lengthscales and outputscales must be positive.")
    nu = None
    if name == "MaternKernel":
        nu = float(getattr(current, "nu", float("nan")))
        if nu not in (0.5, 1.5, 2.5):
            raise NotImplementedError("MaternKernel nu must be 1/2, 3/2, or 5/2.")
    return name, nu, outputscale, lengthscales


def _extract_batched_outcome_transform(transform: Any, m: int):
    if transform is None:
        return np.zeros(m, dtype=float), np.ones(m, dtype=float)
    name = _class_name(transform)
    if name != "Standardize":
        raise NotImplementedError(
            "AutoIA transparent multi-output GP semantics support only no outcome "
            "transform or Standardize; received " + name + "."
        )
    means = getattr(transform, "means", None)
    stdvs = getattr(transform, "stdvs", None)
    if means is None or stdvs is None:
        raise NotImplementedError("Standardize does not expose means/stdvs.")
    offsets = _batched_scalars(means, m, name="outcome mean")
    scales = _batched_scalars(stdvs, m, name="outcome scale")
    if np.any(scales <= 0.0):
        raise ValueError("Outcome Standardize scales must be positive.")
    return offsets, scales


def freeze_gpytorch_exact_gp(model: Any) -> FrozenExactGPState:
    """Freeze the supported stored state of one fitted scalar exact GP.

    This function remains public for expert/audit use, but transparent AutoIA
    execution calls it automatically when a supported fitted model is reached.
    """
    if bool(getattr(model, "training", False)):
        raise NotImplementedError(
            "Transparent fitted-GP semantics require model.eval() so the intended "
            "operation is posterior prediction rather than training-mode prior evaluation."
        )
    train_inputs = getattr(model, "train_inputs", None)
    train_targets = getattr(model, "train_targets", None)
    if not train_inputs or train_targets is None:
        raise NotImplementedError("Supported exact GP must expose train_inputs and train_targets.")
    if len(train_inputs) != 1:
        raise NotImplementedError("Only one input tensor is supported for scalar exact GPs.")
    X = _as_numpy(train_inputs[0])
    y = _as_numpy(train_targets).reshape(-1)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or y.size != X.shape[0]:
        raise NotImplementedError("Only unbatched scalar exact-GP training data are supported.")
    n, d = X.shape

    mean_module = getattr(model, "mean_module", None)
    mean_name = _class_name(mean_module)
    if mean_name not in _SUPPORTED_MEANS:
        raise NotImplementedError(
            "AutoIA transparent GP semantics support ZeroMean or scalar ConstantMean; received " + mean_name + "."
        )
    mean_constant = 0.0 if mean_name == "ZeroMean" else _scalar(getattr(mean_module, "constant"))

    kernel, kind, outputscale = _unwrap_kernel(getattr(model, "covar_module", None))
    lengthscale = _vector(getattr(kernel, "lengthscale"), d)
    if np.any(lengthscale <= 0.0) or outputscale <= 0.0:
        raise ValueError("Kernel lengthscales and outputscale must be positive.")
    nu = None
    if kind == "MaternKernel":
        nu = float(getattr(kernel, "nu", float("nan")))
        if nu not in (0.5, 1.5, 2.5):
            raise NotImplementedError("MaternKernel nu must be 1/2, 3/2, or 5/2.")

    likelihood = getattr(model, "likelihood", None)
    if likelihood is None:
        raise NotImplementedError("Supported exact GP must expose its Gaussian likelihood as model.likelihood.")
    noise_covar = getattr(likelihood, "noise_covar", None)
    noise_obj = getattr(noise_covar, "noise", None) if noise_covar is not None else getattr(likelihood, "noise", None)
    if noise_obj is None:
        raise NotImplementedError("Supported Gaussian likelihood must expose stored observation noise.")
    noise = _as_numpy(noise_obj).reshape(-1)
    if noise.size == 1:
        noise = np.repeat(noise, n)
    if noise.size != n or np.any(noise < 0.0) or np.any(~np.isfinite(noise)):
        raise NotImplementedError("Only scalar or one-value-per-training-point Gaussian noise is supported.")

    input_transform = getattr(model, "input_transform", None)
    input_offset, input_scale = _extract_affine_input_transform(input_transform, d)
    if np.any(input_scale <= 0.0):
        raise ValueError("Affine input-transform scales must be positive.")

    # ExactGP.train_inputs are the coordinates consumed by covar_module.  For
    # ordinary GPyTorch there is no transform.  For BoTorch SingleTaskGP the
    # constructor stores transformed training inputs, while user queries are in
    # physical coordinates and are transformed before model evaluation.
    K = _kernel_numeric(kind, nu, outputscale, lengthscale, X, X)
    K[np.diag_indices_from(K)] += noise
    try:
        L = np.linalg.cholesky(K)
        residual = y - mean_constant
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, residual))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Frozen GP covariance matrix is not numerically SPD.") from exc
    if np.any(~np.isfinite(alpha)) or np.any(~np.isfinite(L)):
        raise ValueError("Frozen GP posterior state is nonfinite.")

    outcome_offset, outcome_scale = _extract_outcome_transform(getattr(model, "outcome_transform", None))
    if outcome_scale <= 0.0:
        raise ValueError("Outcome Standardize scale must be positive.")

    source_kind = "botorch" if _has_botorch_single_task_lineage(type(model)) else "gpytorch"
    return FrozenExactGPState(
        train_x=np.asarray(X, dtype=float).copy(),
        alpha=np.asarray(alpha, dtype=float).copy(),
        chol=np.asarray(L, dtype=float).copy(),
        mean_constant=float(mean_constant), kernel_kind=kind, matern_nu=nu,
        lengthscale=np.asarray(lengthscale, dtype=float).copy(), outputscale=float(outputscale),
        input_offset=np.asarray(input_offset, dtype=float).copy(), input_scale=np.asarray(input_scale, dtype=float).copy(),
        outcome_offset=float(outcome_offset), outcome_scale=float(outcome_scale), source_kind=source_kind,
    )


def freeze_botorch_single_task_gp(
    model: Any,
) -> FrozenExactGPState | FrozenIndependentMultiOutputGPState:
    """Freeze a supported BoTorch ``SingleTaskGP`` posterior state.

    In addition to the scalar-output case, this supports the standard BoTorch
    multi-output construction with no external model batch shape: BoTorch stores
    the independent outputs as the left-most GPyTorch batch dimension.
    """
    if not _has_botorch_single_task_lineage(type(model)):
        raise NotImplementedError("Expected a supported BoTorch SingleTaskGP-like model.")

    num_outputs = int(getattr(model, "num_outputs", 1))
    if num_outputs <= 1:
        return freeze_gpytorch_exact_gp(model)

    if bool(getattr(model, "training", False)):
        raise NotImplementedError(
            "Transparent fitted-GP semantics require model.eval() for posterior prediction."
        )
    # Restrict this first multi-output surface to the common no-model-batch case.
    batch_shape = tuple(getattr(model, "batch_shape", ()))
    if batch_shape:
        raise NotImplementedError(
            "Validated multi-output SingleTaskGP currently supports independent outputs "
            "with no additional model batch dimensions."
        )

    train_inputs = getattr(model, "train_inputs", None)
    train_targets = getattr(model, "train_targets", None)
    if not train_inputs or train_targets is None or len(train_inputs) != 1:
        raise NotImplementedError("Supported exact GP must expose one train_inputs tensor and train_targets.")
    Xb = _as_numpy(train_inputs[0])
    yb = _as_numpy(train_targets)
    if Xb.ndim != 3 or yb.ndim != 2:
        raise NotImplementedError(
            "Expected BoTorch independent-output internal shapes (m,n,d) and (m,n); "
            f"received {Xb.shape} and {yb.shape}."
        )
    m, n, d = Xb.shape
    if m != num_outputs or yb.shape != (m, n):
        raise NotImplementedError(
            "BoTorch independent-output batch dimensions do not match num_outputs."
        )

    mean_module = getattr(model, "mean_module", None)
    mean_name = _class_name(mean_module)
    if mean_name not in _SUPPORTED_MEANS:
        raise NotImplementedError(
            "AutoIA transparent GP semantics support ZeroMean or ConstantMean; received "
            + mean_name + "."
        )
    means = (
        np.zeros(m, dtype=float)
        if mean_name == "ZeroMean"
        else _batched_scalars(getattr(mean_module, "constant"), m, name="mean constant")
    )

    kind, nu, outputscales, lengthscales = _extract_batched_kernel(
        getattr(model, "covar_module", None), m, d
    )

    likelihood = getattr(model, "likelihood", None)
    if likelihood is None:
        raise NotImplementedError("Supported exact GP must expose model.likelihood.")
    noise_covar = getattr(likelihood, "noise_covar", None)
    noise_obj = (
        getattr(noise_covar, "noise", None)
        if noise_covar is not None
        else getattr(likelihood, "noise", None)
    )
    if noise_obj is None:
        raise NotImplementedError("Supported Gaussian likelihood must expose stored observation noise.")
    noise_raw = _as_numpy(noise_obj)
    if noise_raw.size == 1:
        noises = np.full((m, n), float(noise_raw.reshape(-1)[0]), dtype=float)
    elif noise_raw.size == m:
        noises = np.repeat(noise_raw.reshape(m, 1), n, axis=1)
    elif noise_raw.size == m * n:
        noises = noise_raw.reshape(m, n)
    else:
        raise NotImplementedError(
            "Multi-output Gaussian noise must be shared, one value per output, or "
            "one value per output/training point."
        )
    if np.any(noises < 0.0) or np.any(~np.isfinite(noises)):
        raise ValueError("Stored Gaussian noise values must be finite and nonnegative.")

    input_offset, input_scale = _extract_affine_input_transform(
        getattr(model, "input_transform", None), d
    )
    outcome_offsets, outcome_scales = _extract_batched_outcome_transform(
        getattr(model, "outcome_transform", None), m
    )

    states = []
    for j in range(m):
        X = np.asarray(Xb[j], dtype=float)
        y = np.asarray(yb[j], dtype=float)
        K = _kernel_numeric(
            kind, nu, float(outputscales[j]), np.asarray(lengthscales[j], dtype=float), X, X
        )
        K[np.diag_indices_from(K)] += noises[j]
        try:
            L = np.linalg.cholesky(K)
            residual = y - float(means[j])
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, residual))
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"Frozen GP covariance matrix for output {j} is not numerically SPD."
            ) from exc
        states.append(
            FrozenExactGPState(
                train_x=X.copy(), alpha=np.asarray(alpha, dtype=float).copy(),
                chol=np.asarray(L, dtype=float).copy(), mean_constant=float(means[j]),
                kernel_kind=kind, matern_nu=nu,
                lengthscale=np.asarray(lengthscales[j], dtype=float).copy(),
                outputscale=float(outputscales[j]),
                input_offset=np.asarray(input_offset, dtype=float).copy(),
                input_scale=np.asarray(input_scale, dtype=float).copy(),
                outcome_offset=float(outcome_offsets[j]),
                outcome_scale=float(outcome_scales[j]), source_kind="botorch",
            )
        )
    return FrozenIndependentMultiOutputGPState(tuple(states))


def freeze_sklearn_gaussian_process_regressor(model: Any) -> FrozenExactGPState:
    """Freeze one fitted scalar sklearn GaussianProcessRegressor.

    The numerical optimization performed by sklearn is not certified.  AutoIA
    treats the fitted binary64 state as fixed stored data and rigorously encloses
    posterior evaluation conditional on that state.
    """
    if not _has_sklearn_gpr_lineage(type(model)):
        raise NotImplementedError("Expected sklearn GaussianProcessRegressor.")

    required = ("X_train_", "alpha_", "L_", "kernel_")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise NotImplementedError(
            "Transparent sklearn GP semantics require a fitted GaussianProcessRegressor; "
            "missing " + ", ".join(missing) + "."
        )

    X = _as_numpy(getattr(model, "X_train_"))
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise NotImplementedError("Only ordinary 2-D scalar-output sklearn GP inputs are supported.")
    n, d = X.shape

    alpha = _as_numpy(getattr(model, "alpha_"))
    if alpha.ndim == 2 and alpha.shape[1] == 1:
        alpha = alpha[:, 0]
    alpha = alpha.reshape(-1)
    if alpha.size != n:
        raise NotImplementedError(
            "Only scalar-output sklearn GaussianProcessRegressor models are supported."
        )

    L = _as_numpy(getattr(model, "L_"))
    if L.shape != (n, n):
        raise NotImplementedError("sklearn GPR L_ must be an n-by-n Cholesky factor.")
    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(alpha)) or np.any(~np.isfinite(L)):
        raise ValueError("Frozen sklearn GP state contains nonfinite stored values.")
    if np.any(np.diag(L) <= 0.0):
        raise ValueError("Frozen sklearn GP Cholesky factor must have positive diagonal.")

    kernel, kind, outputscale = _unwrap_sklearn_kernel(getattr(model, "kernel_"))
    lengthscale = _vector(getattr(kernel, "length_scale"), d)
    if np.any(lengthscale <= 0.0):
        raise ValueError("Kernel lengthscales must be positive.")
    nu = None
    if kind == "MaternKernel":
        nu = float(getattr(kernel, "nu", float("nan")))
        if nu not in (0.5, 1.5, 2.5):
            raise NotImplementedError("sklearn Matern nu must be 1/2, 3/2, or 5/2.")

    # sklearn GPR uses a zero prior mean internally.  normalize_y=True is an
    # affine output transform applied after posterior evaluation; the fitted
    # alpha_ coefficients already correspond to the normalized training target.
    y_mean = getattr(model, "_y_train_mean", 0.0)
    y_std = getattr(model, "_y_train_std", 1.0)
    outcome_offset = _scalar(y_mean)
    outcome_scale = _scalar(y_std)
    if outcome_scale <= 0.0:
        raise ValueError("sklearn output normalization scale must be positive.")

    return FrozenExactGPState(
        train_x=np.asarray(X, dtype=float).copy(),
        alpha=np.asarray(alpha, dtype=float).copy(),
        chol=np.asarray(L, dtype=float).copy(),
        mean_constant=0.0,
        kernel_kind=kind,
        matern_nu=nu,
        lengthscale=np.asarray(lengthscale, dtype=float).copy(),
        outputscale=float(outputscale),
        input_offset=np.zeros(d, dtype=float),
        input_scale=np.ones(d, dtype=float),
        outcome_offset=float(outcome_offset),
        outcome_scale=float(outcome_scale),
        source_kind="sklearn",
    )


def _transform_query(x, state: FrozenExactGPState):
    arr = np.asarray(x, dtype=object)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != state.dimension:
        raise ValueError(
            f"GP query must have shape (d,) or (n,d) with d={state.dimension}; received {arr.shape}."
        )
    out = np.empty(arr.shape, dtype=object)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            out[i, j] = (arr[i, j] - float(state.input_offset[j])) / float(state.input_scale[j])
    return out


def frozen_gp_posterior(model_x, state: FrozenExactGPState, *, outcome_space: bool = False):
    Xq = _transform_query(model_x, state)
    means = np.empty(Xq.shape[0], dtype=object)
    variances = np.empty(Xq.shape[0], dtype=object)
    for qi in range(Xq.shape[0]):
        k = _kernel_validated(
            state.kernel_kind, state.matern_nu, state.outputscale,
            state.lengthscale, Xq[qi], state.train_x,
        )
        mu = float(state.mean_constant)
        for i, a in enumerate(state.alpha):
            mu = mu + float(a) * k[i]
        # v = L^{-1} k with fixed stored L.  Solving by explicit forward
        # substitution keeps every uncertain arithmetic operation in AutoIA.
        v = np.empty(len(k), dtype=object)
        for i in range(len(k)):
            rhs = k[i]
            for j in range(i):
                rhs = rhs - float(state.chol[i, j]) * v[j]
            v[i] = rhs / float(state.chol[i, i])
        var = float(state.outputscale)
        for vi in v:
            var = var - vi * vi
        if outcome_space:
            mu = float(state.outcome_offset) + float(state.outcome_scale) * mu
            var = (float(state.outcome_scale) ** 2) * var
        means[qi] = mu
        variances[qi] = var
    return means, variances


def frozen_gp_posterior_mean(model_x, state: FrozenExactGPState, *, outcome_space: bool = False):
    return frozen_gp_posterior(model_x, state, outcome_space=outcome_space)[0]


def frozen_gp_posterior_variance(model_x, state: FrozenExactGPState, *, outcome_space: bool = False):
    return frozen_gp_posterior(model_x, state, outcome_space=outcome_space)[1]


def frozen_gp_posterior_covariance(model_x, state: FrozenExactGPState, *, outcome_space: bool = False):
    """Rigorous joint posterior covariance for the supported stationary kernels."""
    Xq = _transform_query(model_x, state)
    nq = Xq.shape[0]
    solved = []
    for qi in range(nq):
        k = _kernel_validated(
            state.kernel_kind, state.matern_nu, state.outputscale,
            state.lengthscale, Xq[qi], state.train_x,
        )
        v = np.empty(len(k), dtype=object)
        for i in range(len(k)):
            rhs = k[i]
            for j in range(i):
                rhs = rhs - float(state.chol[i, j]) * v[j]
            v[i] = rhs / float(state.chol[i, i])
        solved.append(v)

    cov = np.empty((nq, nq), dtype=object)
    scale2 = float(state.outcome_scale) ** 2 if outcome_space else 1.0
    for i in range(nq):
        for j in range(i, nq):
            cij = _kernel_pair_validated(
                state.kernel_kind, state.matern_nu, state.outputscale,
                state.lengthscale, Xq[i], Xq[j],
            )
            for r in range(len(solved[i])):
                cij = cij - solved[i][r] * solved[j][r]
            cij = scale2 * cij
            cov[i, j] = cij
            cov[j, i] = cij
    return cov


class _GPQueryTensor:
    """Tiny tensor-shaped carrier used only for uncertain GP query construction."""
    __slots__ = ("_array", "_dtype")
    def __init__(self, values, dtype=None):
        self._array = np.asarray(values, dtype=object)
        self._dtype = dtype
    @property
    def dtype(self): return self._dtype
    @property
    def shape(self): return self._array.shape
    @property
    def ndim(self): return self._array.ndim
    def dim(self): return self._array.ndim
    def ndimension(self): return self._array.ndim
    @property
    def T(self): return _GPQueryTensor(self._array.T, self._dtype)
    def numpy(self): return self._array.copy()
    def unsqueeze(self, dim): return _GPQueryTensor(np.expand_dims(self._array, int(dim)), self._dtype)
    def squeeze(self, dim=None): return _GPQueryTensor(np.squeeze(self._array, axis=None if dim is None else int(dim)), self._dtype)
    def reshape(self, *shape): return _GPQueryTensor(self._array.reshape(*shape), self._dtype)
    def view(self, *shape): return self.reshape(*shape)
    def flatten(self, *args, **kwargs): return _GPQueryTensor(self._array.flatten(), self._dtype)
    def transpose(self, dim0, dim1): return _GPQueryTensor(np.swapaxes(self._array, int(dim0), int(dim1)), self._dtype)
    def detach(self): return self
    def cpu(self): return self
    def double(self): return self
    def float(self): return self
    def contiguous(self): return self
    def to(self, *args, **kwargs): return self
    def __getitem__(self, key):
        out = self._array[key]
        return _GPQueryTensor(out, self._dtype) if isinstance(out, np.ndarray) else out
    def __len__(self): return len(self._array)


def _contains_validated_query(value: Any) -> bool:
    from ..enclosures import ValidatedValue
    if isinstance(value, ValidatedValue): return True
    if isinstance(value, _GPQueryTensor): return any(isinstance(v, ValidatedValue) for v in value._array.flat)
    if isinstance(value, np.ndarray): return value.dtype == object and any(isinstance(v, ValidatedValue) for v in value.flat)
    if isinstance(value, (list, tuple)): return any(_contains_validated_query(v) for v in value)
    return False


def _query_array(value: Any):
    if isinstance(value, _GPQueryTensor): return value._array
    return np.asarray(value, dtype=object)


def _object_array(value: Any) -> np.ndarray:
    if isinstance(value, _ValidatedTensor):
        return value._array
    if isinstance(value, _GPQueryTensor):
        return value._array
    try:
        return np.asarray(_as_numpy(value), dtype=object)
    except Exception:
        return np.asarray(value, dtype=object)


def _elementwise_unary_array(arr: np.ndarray, op: str) -> np.ndarray:
    out = np.empty(arr.shape, dtype=object)
    for idx in np.ndindex(arr.shape):
        value = arr[idx]
        if hasattr(value, "unary"):
            out[idx] = value.unary(op)
        elif op == "sqrt":
            out[idx] = math.sqrt(float(value))
        elif op == "exp":
            out[idx] = math.exp(float(value))
        elif op == "erf":
            out[idx] = math.erf(float(value))
        else:  # pragma: no cover
            raise NotImplementedError(op)
    return out


def _validated_clamp_scalar(value, *, lower=None, upper=None):
    """Rigorous pointwise clamp for one scalar validated value.

    A clamp is nonsmooth, so this intentionally collapses the scalar to its
    rigorous interval before applying the monotone clamp.  This may lose
    dependence but exactly matches the third-party source semantics.
    """
    from ..enclosures import ValidatedValue
    from ..enclosures.interval import IntervalEnclosure
    from ..trusted import Interval

    if not isinstance(value, ValidatedValue):
        x = float(value)
        if lower is not None:
            x = max(float(lower), x)
        if upper is not None:
            x = min(float(upper), x)
        return x
    I, _ = value.range(value.range_method)
    lo, hi = float(I.lo), float(I.hi)
    if lower is not None:
        lo, hi = max(float(lower), lo), max(float(lower), hi)
    if upper is not None:
        lo, hi = min(float(upper), lo), min(float(upper), hi)
    clipped = Interval(lo, hi)
    rep = IntervalEnclosure(clipped, value.h, value.order, value.backend, value.range_method)
    return ValidatedValue(rep, clipped)


class _ValidatedTensor:
    """Small tensor-shaped carrier for validated third-party tensor algebra.

    This is deliberately not arbitrary Torch support.  It implements the narrow
    deterministic tensor operations used by supported analytic library code once
    uncertain scalar arithmetic has entered the computation.
    """
    __slots__ = ("_array",)
    __array_priority__ = 10000

    def __init__(self, values):
        self._array = np.asarray(values, dtype=object)

    @property
    def shape(self): return self._array.shape
    @property
    def ndim(self): return self._array.ndim
    def dim(self): return self._array.ndim
    def ndimension(self): return self._array.ndim
    def __len__(self): return len(self._array)
    def __repr__(self): return f"_ValidatedTensor(shape={self.shape})"

    def _binary(self, other, op):
        a, b = np.broadcast_arrays(self._array, _object_array(other))
        out = np.empty(a.shape, dtype=object)
        for idx in np.ndindex(a.shape):
            out[idx] = op(a[idx], b[idx])
        return _ValidatedTensor(out)

    def __add__(self, other): return self._binary(other, lambda a,b: a+b)
    def __radd__(self, other): return self._binary(other, lambda a,b: b+a)
    def __sub__(self, other): return self._binary(other, lambda a,b: a-b)
    def __rsub__(self, other): return self._binary(other, lambda a,b: b-a)
    def __mul__(self, other): return self._binary(other, lambda a,b: a*b)
    def __rmul__(self, other): return self._binary(other, lambda a,b: b*a)
    def __truediv__(self, other): return self._binary(other, lambda a,b: a/b)
    def __rtruediv__(self, other): return self._binary(other, lambda a,b: b/a)
    def __pow__(self, power): return self._binary(power, lambda a,b: a**b)
    def __neg__(self): return _ValidatedTensor(-self._array)

    # torch.distributions validates Normal samples using ``value == value``.
    # Validated AutoIA values are finite by construction; return an ordinary
    # boolean Torch tensor solely for this shape/NaN guard.
    def __eq__(self, other):
        try:
            import torch
            return torch.ones(self.shape, dtype=torch.bool)
        except Exception:  # pragma: no cover - Torch is present on this path
            return np.ones(self.shape, dtype=bool)

    def __getitem__(self, key):
        out = self._array[key]
        return _ValidatedTensor(out) if isinstance(out, np.ndarray) else out

    def reshape(self, *shape): return _ValidatedTensor(self._array.reshape(*shape))
    def view(self, *shape): return self.reshape(*shape)
    def unsqueeze(self, dim): return _ValidatedTensor(np.expand_dims(self._array, int(dim)))
    def squeeze(self, dim=None):
        return _ValidatedTensor(np.squeeze(self._array, axis=None if dim is None else int(dim)))
    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        target = []
        src = self.shape
        if len(shape) < len(src):
            raise ValueError("expand target has fewer dimensions than source")
        padded = (1,) * (len(shape) - len(src)) + src
        for want, have in zip(shape, padded):
            want = int(want)
            target.append(have if want == -1 else want)
        arr = self._array.reshape(padded)
        return _ValidatedTensor(np.broadcast_to(arr, tuple(target)))

    def clamp_min(self, value):
        out = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            out[idx] = _validated_clamp_scalar(self._array[idx], lower=value)
        return _ValidatedTensor(out)

    def clamp_max(self, value):
        out = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            out[idx] = _validated_clamp_scalar(self._array[idx], upper=value)
        return _ValidatedTensor(out)

    def sqrt(self): return _ValidatedTensor(_elementwise_unary_array(self._array, "sqrt"))
    def exp(self): return _ValidatedTensor(_elementwise_unary_array(self._array, "exp"))
    def erf(self): return _ValidatedTensor(_elementwise_unary_array(self._array, "erf"))

    def gather(self, dim, index):
        idx = _as_numpy(index, dtype=int)
        return _ValidatedTensor(np.take_along_axis(self._array, idx, axis=int(dim)))

    def prod(self, dim=None):
        if dim is None:
            vals = list(self._array.flat)
            acc = 1.0
            for v in vals: acc = acc * v
            return _ValidatedTensor(np.asarray(acc, dtype=object))
        axis = int(dim)
        return _ValidatedTensor(np.prod(self._array, axis=axis, dtype=object))

    def sum(self, dim=None):
        if dim is None:
            return _ValidatedTensor(np.asarray(np.sum(self._array, dtype=object), dtype=object))
        return _ValidatedTensor(np.sum(self._array, axis=int(dim), dtype=object))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else dict(kwargs)
        name = getattr(func, "__name__", "")
        module = str(getattr(func, "__module__", ""))
        if name == "stack" and module.startswith("torch"):
            tensors = args[0]
            dim = int(kwargs.get("dim", args[1] if len(args) > 1 else 0))
            return cls(np.stack([_object_array(v) for v in tensors], axis=dim))
        if name == "erf" and module.startswith("torch"):
            value = args[0]
            return value.erf() if isinstance(value, cls) else NotImplemented
        return NotImplemented


class _ValidatedNormalProxy:
    """Normal-distribution arithmetic for validated tensor arguments.

    Numeric calls delegate to the original Torch distribution.  Validated calls
    reconstruct only the elementary Normal PDF/CDF formulas using AutoIA-supported
    arithmetic.  This is distribution semantics, not acquisition-function semantics.
    """
    __slots__ = ("_normal", "_loc", "_scale")

    def __init__(self, normal):
        self._normal = normal
        self._loc = _scalar(getattr(normal, "loc"))
        self._scale = _scalar(getattr(normal, "scale"))
        if self._scale <= 0.0:
            raise ValueError("Normal distribution scale must be positive.")

    def log_prob(self, value):
        if not isinstance(value, _ValidatedTensor):
            return self._normal.log_prob(value)
        z = (value - self._loc) / self._scale
        return -0.5 * (z ** 2) - math.log(self._scale) - math.log(math.sqrt(2.0 * math.pi))

    def cdf(self, value):
        if not isinstance(value, _ValidatedTensor):
            return self._normal.cdf(value)
        z = (value - self._loc) / (self._scale * math.sqrt(2.0))
        return 0.5 * (1.0 + z.erf())

    def __getattr__(self, name):
        return getattr(self._normal, name)


def _validated_botorch_posterior_from_state(x, state):
    arr = _query_array(x)
    if isinstance(state, FrozenIndependentMultiOutputGPState):
        d = state.dimension
        if arr.ndim < 2 or arr.shape[-1] != d:
            raise ValueError(
                f"BoTorch GP query must have final dimension {d}; received {arr.shape}."
            )
        leading = arr.shape[:-1]
        flat = arr.reshape(-1, d)
        m = state.num_outputs
        means = np.empty((flat.shape[0], m), dtype=object)
        variances = np.empty((flat.shape[0], m), dtype=object)
        for j, output_state in enumerate(state.outputs):
            mu_j, var_j = frozen_gp_posterior(flat, output_state, outcome_space=True)
            means[:, j] = np.asarray(mu_j, dtype=object).reshape(-1)
            variances[:, j] = np.asarray(var_j, dtype=object).reshape(-1)
        return (
            means.reshape(leading + (m,)),
            variances.reshape(leading + (m,)),
        )

    d = state.dimension
    if arr.ndim < 2 or arr.shape[-1] != d:
        raise ValueError(
            f"BoTorch GP query must have final dimension {d}; received {arr.shape}."
        )
    leading = arr.shape[:-1]
    flat = arr.reshape(-1, d)
    mean, var = frozen_gp_posterior(flat, state, outcome_space=True)
    return (
        np.asarray(mean, dtype=object).reshape(leading + (1,)),
        np.asarray(var, dtype=object).reshape(leading + (1,)),
    )


class _ValidatedPosterior:
    __slots__ = ("mean", "variance", "_botorch")
    def __init__(self, mean, variance, *, botorch: bool):
        if botorch:
            self.mean = _ValidatedTensor(mean)
            self.variance = _ValidatedTensor(variance)
        else:
            self.mean = np.asarray(mean, dtype=object).reshape(-1)
            self.variance = np.asarray(variance, dtype=object).reshape(-1)
        self._botorch = bool(botorch)
    @property
    def stddev(self):
        if isinstance(self.variance, _ValidatedTensor):
            return self.variance.sqrt()
        out = np.empty(self.variance.shape, dtype=object)
        for idx in np.ndindex(out.shape): out[idx] = _sqrt_value(self.variance[idx])
        return out
    @property
    def covariance_matrix(self):
        raise NotImplementedError(
            "Transparent GP covariance_matrix semantics are not yet supported; "
            "use posterior.mean or posterior.variance for the supported scalar-query boundary."
        )


def _sqrt_sklearn_clipped_variance(value):
    """Rigorous sklearn return_std semantics: sqrt(max(variance, 0)).

    sklearn itself clips negative numerical predictive variances to zero before
    taking the square root.  For an uncertain AutoIA value we enclose that
    pointwise clipping by first collapsing to its rigorous scalar interval.
    This may be loose, but it is sound and fail-safe.
    """
    from ..enclosures import ValidatedValue
    from ..enclosures.interval import IntervalEnclosure
    from ..trusted import Interval

    if not isinstance(value, ValidatedValue):
        return math.sqrt(max(float(value), 0.0))
    I, _ = value.range(value.range_method)
    clipped = Interval(max(0.0, float(I.lo)), max(0.0, float(I.hi)))
    rep = IntervalEnclosure(
        clipped, value.h, value.order, value.backend, value.range_method
    )
    clipped_value = ValidatedValue(rep, clipped)
    return clipped_value.unary("sqrt")


def _validated_sklearn_predict(model: Any, x, args, kwargs):
    if not _contains_validated_query(x):
        return None
    if args:
        raise NotImplementedError(
            "Validated sklearn GPR predict accepts X plus keyword return_std/return_cov only."
        )
    kw = dict(kwargs)
    return_std = bool(kw.pop("return_std", False))
    return_cov = bool(kw.pop("return_cov", False))
    if kw:
        raise NotImplementedError(
            "Unsupported sklearn GaussianProcessRegressor.predict keyword(s): "
            + ", ".join(sorted(kw))
        )
    if return_std and return_cov:
        raise RuntimeError("At most one of return_std or return_cov can be requested.")
    state = freeze_sklearn_gaussian_process_regressor(model)
    mean, var = frozen_gp_posterior(
        _query_array(x), state, outcome_space=True,
    )
    mean = np.asarray(mean, dtype=object).reshape(-1)
    if return_cov:
        cov = frozen_gp_posterior_covariance(
            _query_array(x), state, outcome_space=True,
        )
        return mean, cov
    if not return_std:
        return mean
    std = np.empty(np.asarray(var, dtype=object).shape, dtype=object)
    for idx in np.ndindex(std.shape):
        std[idx] = _sqrt_sklearn_clipped_variance(np.asarray(var, dtype=object)[idx])
    return mean, std.reshape(-1)


class TransparentExactGPProxy:
    """Delegate numeric calls, substitute validated semantics for uncertain queries."""
    __slots__ = ("_model", "_state")
    def __init__(self, model):
        self._model = model
        self._state = None
    def _frozen(self):
        if self._state is None:
            if _has_sklearn_gpr_lineage(type(self._model)):
                self._state = freeze_sklearn_gaussian_process_regressor(self._model)
            else:
                self._state = freeze_gpytorch_exact_gp(self._model)
        return self._state
    def __call__(self, x, *args, **kwargs):
        if not _contains_validated_query(x):
            return self._model(x, *args, **kwargs)
        if args or kwargs:
            raise NotImplementedError("Validated GP model calls currently accept only the query tensor.")
        mean, var = frozen_gp_posterior(_query_array(x), self._frozen(), outcome_space=False)
        return _ValidatedPosterior(mean, var, botorch=False)
    def posterior(self, x, *args, **kwargs):
        if not _contains_validated_query(x):
            return self._model.posterior(x, *args, **kwargs)
        kw = dict(kwargs)
        observation_noise = kw.pop("observation_noise", False)
        posterior_transform = kw.pop("posterior_transform", None)
        if args or kw or observation_noise not in (False, None) or posterior_transform is not None:
            raise NotImplementedError(
                "Validated BoTorch posterior currently supports latent posterior(x) only "
                "with observation_noise=False and posterior_transform=None."
            )
        state = freeze_botorch_single_task_gp(self._model)
        mean, var = _validated_botorch_posterior_from_state(x, state)
        return _ValidatedPosterior(mean, var, botorch=True)
    def predict(self, x, *args, **kwargs):
        if not _contains_validated_query(x):
            return self._model.predict(x, *args, **kwargs)
        out = _validated_sklearn_predict(self._model, x, args, kwargs)
        if out is None:  # pragma: no cover - guarded above
            return self._model.predict(x, *args, **kwargs)
        return out
    def __getattr__(self, name): return getattr(self._model, name)


def looks_like_supported_gp_object(value: Any) -> bool:
    """Return True only for the narrow fitted exact-GP boundary AutoIA supports.

    User-defined GPyTorch models normally subclass ``gpytorch.models.ExactGP``.
    Their concrete class therefore lives in the user's module (often
    ``__main__``), not in a module whose name starts with ``gpytorch``.  The
    recognizer must inspect the inheritance chain rather than only the concrete
    class module.  This remains fail-closed: only an ExactGP or SingleTaskGP
    lineage plus the required fitted-state attributes is accepted.
    """
    if isinstance(value, TransparentExactGPProxy):
        return True

    cls = type(value)
    has_gpytorch_exactgp_base = _has_gpytorch_exactgp_lineage(cls)
    has_botorch_single_task_base = _has_botorch_single_task_lineage(cls)
    has_sklearn_gpr_base = _has_sklearn_gpr_lineage(cls)
    structural_test_marker = bool(getattr(value, "__autoia_supported_exact_gp__", False))

    if has_sklearn_gpr_base:
        return bool(
            hasattr(value, "X_train_") and hasattr(value, "alpha_")
            and hasattr(value, "L_") and hasattr(value, "kernel_")
        )

    if not (has_gpytorch_exactgp_base or has_botorch_single_task_base or structural_test_marker):
        return False

    return bool(
        hasattr(value, "train_inputs") and hasattr(value, "train_targets")
        and hasattr(value, "mean_module") and hasattr(value, "covar_module")
        and hasattr(value, "likelihood")
    )


def maybe_wrap_gp_object(value: Any):
    return TransparentExactGPProxy(value) if looks_like_supported_gp_object(value) else value


_INTERCEPTING_GP_CLASS_CACHE: dict[type, type] = {}


def _validated_gpytorch_model_call(model: Any, x, args, kwargs):
    """Numeric fallthrough; validated posterior replacement for ExactGP.__call__."""
    if not _contains_validated_query(x):
        return None
    if args or kwargs:
        raise NotImplementedError(
            "Validated GPyTorch model calls currently accept only the query tensor."
        )
    mean, var = frozen_gp_posterior(
        _query_array(x), freeze_gpytorch_exact_gp(model), outcome_space=False
    )
    return _ValidatedPosterior(mean, var, botorch=False)


def _validated_botorch_posterior_call(model: Any, x, args, kwargs):
    """Validated replacement for the supported SingleTaskGP.posterior boundary."""
    if not _contains_validated_query(x):
        return None
    kw = dict(kwargs)
    observation_noise = kw.pop("observation_noise", False)
    posterior_transform = kw.pop("posterior_transform", None)
    if args or kw or observation_noise not in (False, None) or posterior_transform is not None:
        raise NotImplementedError(
            "Validated BoTorch posterior currently supports latent posterior(x) only "
            "with observation_noise=False and posterior_transform=None."
        )
    state = freeze_botorch_single_task_gp(model)
    mean, var = _validated_botorch_posterior_from_state(x, state)
    return _ValidatedPosterior(mean, var, botorch=True)


def make_intercepting_gp_class(cls: Any):
    """Return a private subclass that preserves numeric training but intercepts
    uncertain posterior queries.

    This is the key semantic used when a GP is *constructed inside* the
    engineering function.  The replacement class exists only in AutoIA's shadow
    execution namespace.  It is still an actual subclass of the user's real
    GPyTorch/BoTorch class, so optimizers, autograd, MLL construction, ``train`` /
    ``eval`` and ordinary numeric calls continue through the third-party library.
    """
    if not isinstance(cls, type):
        return cls
    is_botorch = _has_botorch_single_task_lineage(cls)
    is_gpytorch = _has_gpytorch_exactgp_lineage(cls)
    is_sklearn = _has_sklearn_gpr_lineage(cls)
    if not (is_botorch or is_gpytorch or is_sklearn):
        return cls
    cached = _INTERCEPTING_GP_CLASS_CACHE.get(cls)
    if cached is not None:
        return cached

    if is_botorch:
        class _AutoIAInterceptedGP(cls):  # type: ignore[misc, valid-type]
            def posterior(self, x, *args, **kwargs):
                out = _validated_botorch_posterior_call(self, x, args, kwargs)
                if out is not None:
                    return out
                return super().posterior(x, *args, **kwargs)
    elif is_sklearn:
        class _AutoIAInterceptedGP(cls):  # type: ignore[misc, valid-type]
            def predict(self, x, *args, **kwargs):
                out = _validated_sklearn_predict(self, x, args, kwargs)
                if out is not None:
                    return out
                return super().predict(x, *args, **kwargs)
    else:
        class _AutoIAInterceptedGP(cls):  # type: ignore[misc, valid-type]
            def __call__(self, x, *args, **kwargs):
                out = _validated_gpytorch_model_call(self, x, args, kwargs)
                if out is not None:
                    return out
                return super().__call__(x, *args, **kwargs)

    # Preserve normal-looking introspection in third-party code.  The class is
    # private to one shadow execution namespace and is never installed globally.
    _AutoIAInterceptedGP.__name__ = getattr(cls, "__name__", _AutoIAInterceptedGP.__name__)
    _AutoIAInterceptedGP.__qualname__ = getattr(cls, "__qualname__", _AutoIAInterceptedGP.__qualname__)
    _AutoIAInterceptedGP.__module__ = getattr(cls, "__module__", _AutoIAInterceptedGP.__module__)
    _INTERCEPTING_GP_CLASS_CACHE[cls] = _AutoIAInterceptedGP
    return _AutoIAInterceptedGP


_INTERCEPTING_BOTORCH_CLASS_CACHE: dict[type, type] = {}


def _is_botorch_acquisition_class(cls: Any) -> bool:
    return isinstance(cls, type) and str(getattr(cls, "__module__", "")).startswith(
        "botorch.acquisition"
    )


def make_intercepting_botorch_acquisition_class(cls: Any):
    """Private subclass that replaces fixed Normal helper objects after init.

    BoTorch analytic acquisitions commonly keep a ``torch.distributions.Normal``
    helper in object state.  Torch's distribution methods require their argument
    to be an actual ``torch.Tensor`` and therefore reject AutoIA's validated
    tensor carrier before any arithmetic occurs.  Replacing only that fixed helper
    with ``_ValidatedNormalProxy`` lets the original acquisition ``forward`` code
    execute unchanged while keeping the mathematical operation generic.
    """
    if not _is_botorch_acquisition_class(cls):
        return cls
    cached = _INTERCEPTING_BOTORCH_CLASS_CACHE.get(cls)
    if cached is not None:
        return cached

    class _AutoIAInterceptedAcquisition(cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            normal = getattr(self, "normal", None)
            if normal is not None and _class_name(normal) == "Normal":
                self.normal = _ValidatedNormalProxy(normal)

    _AutoIAInterceptedAcquisition.__name__ = getattr(
        cls, "__name__", _AutoIAInterceptedAcquisition.__name__
    )
    _AutoIAInterceptedAcquisition.__qualname__ = getattr(
        cls, "__qualname__", _AutoIAInterceptedAcquisition.__qualname__
    )
    _AutoIAInterceptedAcquisition.__module__ = getattr(
        cls, "__module__", _AutoIAInterceptedAcquisition.__module__
    )
    _INTERCEPTING_BOTORCH_CLASS_CACHE[cls] = _AutoIAInterceptedAcquisition
    return _AutoIAInterceptedAcquisition


class _GPModelsModuleProxy:
    """Proxy only the GP model classes that need local-construction semantics."""
    __slots__ = ("_target", "_kind")
    def __init__(self, target, kind: str):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_kind", str(kind))
    def __getattr__(self, name):
        value = getattr(object.__getattribute__(self, "_target"), name)
        kind = object.__getattribute__(self, "_kind")
        if (kind == "gpytorch" and name == "ExactGP") or (
            kind == "botorch" and name == "SingleTaskGP"
        ):
            return make_intercepting_gp_class(value)
        return value
    def __setattr__(self, name, value):
        raise AttributeError("AutoIA GP module proxy is read-only.")


class GPModuleProxy:
    """Read-only top-level GPyTorch/BoTorch proxy for local model construction."""
    __slots__ = ("_target", "_kind", "_models_proxy")
    def __init__(self, target, kind: str):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_kind", str(kind))
        models = getattr(target, "models", None)
        object.__setattr__(
            self, "_models_proxy",
            _GPModelsModuleProxy(models, kind) if models is not None else None,
        )
    @property
    def __name__(self):
        return getattr(object.__getattribute__(self, "_target"), "__name__", "")
    def __getattr__(self, name):
        kind = object.__getattribute__(self, "_kind")
        if name == "models":
            proxy = object.__getattribute__(self, "_models_proxy")
            if proxy is not None:
                return proxy
        value = getattr(object.__getattribute__(self, "_target"), name)
        if kind == "sklearn" and name == "GaussianProcessRegressor":
            return make_intercepting_gp_class(value)
        return value
    def __setattr__(self, name, value):
        raise AttributeError("AutoIA GP module proxy is read-only.")


def maybe_make_gp_construction_proxy(value: Any):
    """Replace GP modules/classes only inside AutoIA's private shadow namespace."""
    if isinstance(value, GPModuleProxy):
        return value
    if isinstance(value, type):
        gp_cls = make_intercepting_gp_class(value)
        if gp_cls is not value:
            return gp_cls
        return make_intercepting_botorch_acquisition_class(value)
    name = str(getattr(value, "__name__", ""))
    if name == "gpytorch":
        return GPModuleProxy(value, "gpytorch")
    if name == "botorch":
        return GPModuleProxy(value, "botorch")
    if name in {"sklearn.gaussian_process", "sklearn.gaussian_process._gpr"}:
        return GPModuleProxy(value, "sklearn")
    return value


class TorchQueryProxy:
    """Read-only torch proxy intercepting only uncertain query construction."""
    __slots__ = ("_target",)
    def __init__(self, target): object.__setattr__(self, "_target", target)
    def tensor(self, data, *args, **kwargs):
        if _contains_validated_query(data):
            return _GPQueryTensor(data, kwargs.get("dtype"))
        return self._target.tensor(data, *args, **kwargs)
    def as_tensor(self, data, *args, **kwargs):
        if _contains_validated_query(data): return _GPQueryTensor(data, kwargs.get("dtype"))
        return self._target.as_tensor(data, *args, **kwargs)
    def stack(self, tensors, dim=0, *args, **kwargs):
        if _contains_validated_query(tensors):
            arrays = [_query_array(v) for v in tensors]
            dtype = next((getattr(v, "dtype", None) for v in tensors if getattr(v, "dtype", None) is not None), None)
            return _GPQueryTensor(np.stack(arrays, axis=int(dim)), dtype)
        return self._target.stack(tensors, dim=dim, *args, **kwargs)
    def cat(self, tensors, dim=0, *args, **kwargs):
        if _contains_validated_query(tensors):
            arrays = [_query_array(v) for v in tensors]
            dtype = next((getattr(v, "dtype", None) for v in tensors if getattr(v, "dtype", None) is not None), None)
            return _GPQueryTensor(np.concatenate(arrays, axis=int(dim)), dtype)
        return self._target.cat(tensors, dim=dim, *args, **kwargs)
    def __getattr__(self, name): return getattr(self._target, name)
    def __setattr__(self, name, value): raise AttributeError("AutoIA Torch query proxy is read-only.")


def maybe_make_torch_proxy(value: Any):
    if getattr(value, "__name__", None) == "torch": return TorchQueryProxy(value)
    return value


__all__ = [
    "FrozenExactGPState", "FrozenIndependentMultiOutputGPState",
    "freeze_gpytorch_exact_gp", "freeze_botorch_single_task_gp",
    "freeze_sklearn_gaussian_process_regressor",
    "frozen_gp_posterior", "frozen_gp_posterior_mean", "frozen_gp_posterior_variance",
    "frozen_gp_posterior_covariance",
    "TransparentExactGPProxy", "TorchQueryProxy", "GPModuleProxy",
    "looks_like_supported_gp_object", "make_intercepting_gp_class",
    "maybe_wrap_gp_object", "maybe_make_torch_proxy", "maybe_make_gp_construction_proxy",
]
