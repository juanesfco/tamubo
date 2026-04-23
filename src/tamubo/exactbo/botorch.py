from __future__ import annotations

try:
    import torch
    from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
except ModuleNotFoundError as exc:
    raise ImportError(
        "BoTorch and its dependencies are required to use tamubo.exactbo.botorch. "
        "Please install BoTorch and try again."
    ) from exc

from .torch_partition import exactbo_torch_partitioning

__all__ = ["optimize_acqf_exactbo"]

_SUPPORTED_ACQF_TYPES = (ExpectedImprovement, LogExpectedImprovement)


def optimize_acqf_exactbo(
    acq_function,
    bounds: torch.Tensor,
    epsilon_X,
    epsilon_ei: float,
    max_partitions: int,
    *,
    return_log: bool = False,
    validation: bool = True,
    verbose: bool = False,
    logMask: bool = False,
):
    """
    Optimize a BoTorch EI-style acquisition function with ExactBO.

    This adapter currently supports single-point analytic
    ``ExpectedImprovement`` and ``LogExpectedImprovement`` with
    ``maximize=False``.
    """
    if not isinstance(acq_function, _SUPPORTED_ACQF_TYPES):
        raise TypeError(
            "optimize_acqf_exactbo currently supports BoTorch ExpectedImprovement "
            "and LogExpectedImprovement acquisition functions."
        )

    maximize = bool(getattr(acq_function, "maximize", True))
    if maximize:
        raise ValueError("optimize_acqf_exactbo currently supports only minimize-mode EI acquisition functions (maximize=False).")

    if not hasattr(acq_function, "best_f"):
        raise TypeError("The provided acquisition function does not expose a best_f attribute.")

    best_f = torch.as_tensor(acq_function.best_f).reshape(-1)
    if best_f.numel() != 1:
        raise ValueError("optimize_acqf_exactbo requires a scalar best_f.")

    model = acq_function.model
    if hasattr(model, "eval"):
        model.eval()

    record_log = bool(logMask or return_log)
    result = exactbo_torch_partitioning(
        model,
        bounds,
        epsilon_X,
        epsilon_ei,
        float(best_f[0].item()),
        max_partitions,
        validation=validation,
        verbose=verbose,
        logMask=record_log,
    )

    candidate = result.candidate.unsqueeze(0)
    with torch.no_grad():
        acq_value = acq_function(candidate.unsqueeze(-2)).reshape(-1)[0]

    if return_log:
        return candidate, acq_value, result.log
    return candidate, acq_value
