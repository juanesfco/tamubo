"""Stable package-facing AutoIA entry points.

The active numerical implementation now lives in extracted trusted, polynomial,
enclosure, semantic, and controller layers.  This facade owns stable input and
failure semantics without changing those numerical algorithms.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
import math
import traceback
import time
import numpy as np

from .controller import (
    autoia as _controller_autoia,
    autoia_adaptive_subdivide as _controller_adaptive_subdivide,
    autoia_subdivide as _controller_subdivide,
    RangePolicy as _ControllerRangePolicy,
    SubdivisionPolicy as _ControllerSubdivisionPolicy,
    BranchPolicy as _ControllerBranchPolicy,
    ConfigurationPolicy as _ControllerConfigurationPolicy,
    AutoIAConfiguredResult as _ControllerConfiguredResult,
    configuration_candidates as _configuration_candidates,
    automatic_escalation_candidates as _automatic_escalation_candidates,
    select_dominating_result as _select_dominating_result,
    rigorous_intersection_bounds as _rigorous_intersection_bounds,
    structural_summary as _configuration_structural_summary,
)
from .trusted import Interval as _TrustedInterval, ElementaryDomainError
from .intervals import Interval, attach_input_specification
from .backends import RigorousBinary64Backend
from .enclosures import DomainRefinementRequired
from .policies import RangePolicy, SubdivisionPolicy, BranchPolicy, ConfigurationPolicy
from .results import AutoIAResult, AdaptiveSubdivisionResult, PiecewiseResult, AutoIAPathwiseResult, AutoIAConfiguredResult
from .exceptions import (
    AutoIAError,
    InvalidInputError,
    UnsupportedOperationError,
    CertificationError,
    ResourceLimitError,
    NumericalLimitError,
    ModelExecutionError,
    InternalAutoIAError,
)


_RANGE_INTERNAL = {"termwise", "monotonicity", "bernstein", "portfolio"}
_RANGE_FINAL = _RANGE_INTERNAL | {"adaptive"}
_REPRESENTATIONS = {"polynomial", "interval", "generator"}


def _traceback_frames(exc: BaseException):
    tb = exc.__traceback__
    frames = []
    while tb is not None:
        frames.append(tb.tb_frame)
        tb = tb.tb_next
    return frames


def _frame_is_autoia_internal(frame) -> bool:
    module_name = str(frame.f_globals.get("__name__", ""))
    return (
        module_name.startswith("autoia.trusted")
        or module_name.startswith("autoia.polynomial")
        or module_name.startswith("autoia.enclosures")
        or module_name.startswith("autoia.semantic")
        or module_name.startswith("autoia.controller")
    )


def _last_frame_is_autoia_internal(exc: BaseException) -> bool:
    frames = _traceback_frames(exc)
    return bool(frames and _frame_is_autoia_internal(frames[-1]))


def _internal_frame_after_guard(exc: BaseException) -> bool:
    """Return True when AutoIA reference code was entered *inside* user model code.

    The outer AutoIA controller frame necessarily appears before the model
    callback.  We care about AutoIA internal frames after ``_guarded_model`` because
    they indicate a validated operator/semantic primitive raised the exception,
    rather than the user's ordinary Python code raising on its own.
    """
    frames = _traceback_frames(exc)
    guard_index = None
    for i, frame in enumerate(frames):
        if frame.f_code.co_name == "_guarded_model" and frame.f_code.co_filename == __file__:
            guard_index = i
    if guard_index is None:
        return False
    return any(
        _frame_is_autoia_internal(frame)
        for frame in frames[guard_index + 1 :]
    )


def _looks_like_unsupported_operation(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = (
        "validatedvalue",
        "no callable sin method",
        "no callable cos method",
        "no callable exp method",
        "no callable log method",
        "no callable sqrt method",
        "ufunc",
        "nonnegative integer powers",
        "current solve primitive requires",
        "requires an affine retained polynomial",
        "piecewise values is not yet supported",
        "piecewise validated values",
        "does not yet combine the structural range controller with uncertain python control flow",
    )
    return isinstance(exc, NotImplementedError) or any(m in text for m in markers)


def _guard_model(model: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Preserve AutoIA proof failures while typing exceptions raised by user code."""

    def _guarded_model(p):
        try:
            return model(p)
        except AutoIAError:
            raise
        except Exception as exc:
            # A validated operator or semantic primitive raised while executing
            # the model.  Let the outer internal-exception translator classify it.
            if _internal_frame_after_guard(exc):
                raise
            if _looks_like_unsupported_operation(exc):
                raise UnsupportedOperationError(
                    str(exc),
                    context={"operation_context": "model execution"},
                    suggestions=(
                        "Rewrite the expression using currently supported AutoIA operations.",
                        "Check the documented supported-language boundary.",
                    ),
                ) from exc
            raise ModelExecutionError(
                f"The engineering model raised {type(exc).__name__}: {exc}",
                context={"original_exception_type": type(exc).__name__},
            ) from exc

    _guarded_model.__name__ = getattr(model, "__name__", "_autoia_model")
    _guarded_model.__doc__ = getattr(model, "__doc__", None)

    # Local semantic dispatch must prepare the *user model*, not merely this
    # exception-typing wrapper (whose model reference lives in a closure).
    # Expose a generic preparation hook so semantic.dispatch can preserve the
    # guard without importing the API layer.
    def _prepare_dispatch(traced_solve):
        from .semantic.dispatch import prepare_model_for_dispatch
        prepared_user_model = prepare_model_for_dispatch(model, traced_solve)
        return _guard_model(prepared_user_model)

    _guarded_model.__autoia_prepare_dispatch__ = _prepare_dispatch
    return _guarded_model


def _translate_internal_exception(exc: Exception) -> AutoIAError:
    """Map current reference-core failures to the frozen public taxonomy."""
    if isinstance(exc, AutoIAError):
        return exc

    text = str(exc)
    lower = text.lower()

    if isinstance(exc, NotImplementedError) or _looks_like_unsupported_operation(exc):
        return UnsupportedOperationError(
            text,
            suggestions=(
                "Use operations in AutoIA's supported execution subset.",
                "Use a supported semantic primitive or rewrite with supported generic algebra.",
            ),
        )

    if isinstance(exc, DomainRefinementRequired):
        proof = exc.proof
        return CertificationError(
            text,
            context={
                "proof_obligation": "elementary-function real domain",
                "operation": proof.requirement.operation,
                "domain_status": proof.status,
                "methods_tried": tuple(proof.methods_tried),
            },
            suggestions=(
                "Use autoia(...) so the path/domain controller may refine the physical box.",
                "Increase the domain-refinement budget when the model is known to be real-valued.",
            ),
        )

    if isinstance(exc, ElementaryDomainError):
        return CertificationError(
            text,
            context={"proof_obligation": "elementary-function real domain"},
            suggestions=(
                "Subdivide the input box so it lies inside the function domain.",
                "Check the engineering model's real-domain assumptions.",
            ),
        )

    if isinstance(exc, ZeroDivisionError):
        return CertificationError(
            text,
            context={"proof_obligation": "denominator separated from zero"},
            suggestions=(
                "Subdivide the input box.",
                "Use a stronger representation or proof range method if available.",
            ),
        )

    if isinstance(exc, (OverflowError, FloatingPointError)):
        return NumericalLimitError(
            text,
            context={"original_exception_type": type(exc).__name__},
        )

    if isinstance(exc, np.linalg.LinAlgError):
        return CertificationError(
            f"Nominal/semantic linear-algebra step failed: {text}",
            context={"proof_obligation": "linear-system proposal/certificate"},
            suggestions=(
                "Check whether the center system is singular or ill-conditioned.",
                "Subdivide or choose a different certifier/preconditioner.",
            ),
        )

    if isinstance(exc, RuntimeError):
        if any(
            marker in lower
            for marker in (
                "validated root_scalar",
                "validated fsolve",
                "root_scalar local contraction",
                "fsolve local contraction",
                "implicit predictor",
                "implicit jacobian",
                "unique root",
                "validated linprog",
            )
        ):
            obligation = (
                "linear-program optimal-value certificate"
                if "validated linprog" in lower
                else "implicit nonlinear solve certificate"
            )
            return CertificationError(
                text,
                context={"proof_obligation": obligation},
                suggestions=(
                    "Subdivide the physical input box.",
                    "For root_scalar, tighten the supplied bracket when physically justified.",
                    "For fsolve, use an initial guess nearer the intended nonsingular solution branch.",
                ),
            )
        if any(
            marker in lower
            for marker in (
                "contraction failed",
                "rho_upper",
                "contraction margin lost",
            )
        ):
            return CertificationError(
                text,
                context={"proof_obligation": "contraction certificate"},
                suggestions=(
                    "Subdivide the input box.",
                    "Switch representation or certifier if one is available.",
                ),
            )
        if "budget" in lower or "resource" in lower:
            return ResourceLimitError(text)
        if any(
            marker in lower
            for marker in (
                "disjoint enclosures",
                "internal error",
                "internal soundness error",
                "produced no branches",
                "binary split did not produce",
            )
        ):
            return InternalAutoIAError(
                text, context={"original_exception_type": "RuntimeError"}
            )
        # An unclassified RuntimeError originating in AutoIA internal code is not
        # safe to reinterpret as an ordinary certification failure.
        if _last_frame_is_autoia_internal(exc):
            return InternalAutoIAError(
                text, context={"original_exception_type": "RuntimeError"}
            )

    if isinstance(exc, (ValueError, TypeError)):
        # These messages describe malformed calls to a recognized model-level
        # operation rather than a failed mathematical proof.
        if any(
            marker in lower
            for marker in (
                "a must be square",
                "right-hand side",
                "a and b dimensions do not match",
                "domains/backends do not match",
                "all feasible execution paths must return the same output shape",
                "all feasible execution paths must return the same number of outputs",
                "regulargridinterpolator query leaves the fixed grid",
                "regulargridinterpolator reaches an out-of-bounds path",
            )
        ):
            return ModelExecutionError(
                text, context={"original_exception_type": type(exc).__name__}
            )
        # Public input/configuration checks should have caught these before
        # entering the reference core.  If they still occur, keep the failure
        # typed as invalid input rather than leaking a built-in exception.
        if any(
            marker in lower
            for marker in (
                "at least one uncertain parameter",
                "representation must be",
                "subdivision counts",
                "split_counts",
                "invalid internal_method",
                "invalid final_method",
                "bernstein_max_coefficients",
                "requires finite inputs",
                "finite nonnegative parameter half-widths",
                "pass either range_policy",
            )
        ):
            return InvalidInputError(text)

    if isinstance(exc, AssertionError):
        return InternalAutoIAError(
            text or "An internal AutoIA assertion failed.",
            context={"original_exception_type": "AssertionError"},
        )

    # Unknown exceptions raised from AutoIA internal implementation code are bugs until
    # deliberately classified.  Unknown exceptions from user code should have
    # been wrapped by _guard_model, but retain a defensive ModelExecution fallback.
    if _last_frame_is_autoia_internal(exc):
        return InternalAutoIAError(
            f"Unexpected AutoIA-internal {type(exc).__name__}: {text}",
            context={"original_exception_type": type(exc).__name__},
        )
    return ModelExecutionError(
        f"The engineering model raised {type(exc).__name__}: {text}",
        context={"original_exception_type": type(exc).__name__},
    )


def _invoke_internal(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except AutoIAError:
        raise
    except Exception as exc:
        mapped = _translate_internal_exception(exc)
        raise mapped from exc


def _validate_common(
    model,
    parameter_intervals,
    *,
    order,
    backend,
    representation,
    range_policy,
    range_method=None,
    internal_range_method=None,
    final_range_method=None,
):
    if not callable(model):
        raise InvalidInputError("model must be callable.")
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or int(order) < 0:
        raise InvalidInputError("order must be a nonnegative integer.", context={"order": order})
    if representation not in _REPRESENTATIONS:
        raise InvalidInputError(
            "representation must be 'polynomial', 'interval', or 'generator'.",
            context={"representation": representation},
        )
    try:
        intervals = list(parameter_intervals)
    except Exception as exc:
        raise InvalidInputError("parameter_intervals must be an iterable of Interval objects.") from exc
    if not intervals:
        raise InvalidInputError("At least one uncertain parameter is required.")
    for i, I in enumerate(intervals):
        if not isinstance(I, _TrustedInterval):
            raise InvalidInputError(
                "Each parameter interval must be an autoia.Interval.",
                context={"index": i, "received_type": type(I).__name__},
            )
        if not (math.isfinite(float(I.lo)) and math.isfinite(float(I.hi))):
            raise InvalidInputError(
                "Version-1 candidate contract requires finite interval endpoints.",
                context={"index": i, "lo": I.lo, "hi": I.hi},
            )
    if backend is not None and not isinstance(backend, RigorousBinary64Backend):
        raise InvalidInputError(
            "backend must be RigorousBinary64Backend or None in the public API.",
            context={"received_type": type(backend).__name__},
        )
    if range_policy is not None and any(
        v is not None for v in (range_method, internal_range_method, final_range_method)
    ):
        raise InvalidInputError(
            "Pass either range_policy or the range_method/internal/final options, not both."
        )
    if range_policy is not None and not isinstance(range_policy, _ControllerRangePolicy):
        raise InvalidInputError("range_policy must be an autoia.RangePolicy.")
    if range_method is not None and range_method not in _RANGE_INTERNAL:
        raise InvalidInputError(f"Unknown range method {range_method!r}.")
    if internal_range_method is not None and internal_range_method not in _RANGE_INTERNAL:
        raise InvalidInputError(f"Unknown internal range method {internal_range_method!r}.")
    if final_range_method is not None and final_range_method not in _RANGE_FINAL:
        raise InvalidInputError(f"Unknown final range method {final_range_method!r}.")
    return intervals



def _run_one_configuration(
    model,
    intervals,
    *,
    order,
    backend,
    representation,
    range_policy,
    range_method,
    internal_range_method,
    final_range_method,
    branch_policy,
):
    """Run one independently rigorous configuration through the frozen API guard."""
    return _invoke_internal(
        _controller_autoia,
        _guard_model(model),
        intervals,
        order=int(order),
        range_method=range_method,
        backend=backend,
        representation=representation,
        internal_range_method=internal_range_method,
        final_range_method=final_range_method,
        range_policy=range_policy,
        branch_policy=branch_policy,
    )


def _finalize_configuration_attempts(
    successes,
    attempts,
    *,
    configuration_policy,
    baseline_representation,
    baseline_order,
    selector_signals=None,
):
    """Form the rigorous componentwise intersection and common diagnostics."""
    if not successes:
        raise InternalAutoIAError("Automatic configuration portfolio produced no successful attempts.")

    selected_success_index, selected = _select_dominating_result(successes)
    intersection_lower, intersection_upper = _rigorous_intersection_bounds(successes)
    selected_attempt_index = next(
        i for i, attempt in enumerate(attempts)
        if attempt.get("success") and attempt.get("success_index") == selected_success_index
    )
    nondominated_attempts = []
    for i, attempt in enumerate(attempts):
        if not attempt.get("success"):
            continue
        candidate = successes[int(attempt["success_index"])]
        dominated = any(
            _select_dominating_result([candidate, other])[1] is other
            and other is not candidate
            for other in successes
        )
        if not dominated:
            nondominated_attempts.append(i)

    diagnostics = {
        "mode": configuration_policy.mode,
        "baseline_representation": baseline_representation,
        "baseline_order": int(baseline_order),
        "attempts": attempts,
        "selected_attempt_index": int(selected_attempt_index),
        "selected_representation": attempts[selected_attempt_index]["representation"],
        "selected_order": int(attempts[selected_attempt_index]["order"]),
        "nondominated_attempt_indices": tuple(int(i) for i in nondominated_attempts),
        "intersection_lower": intersection_lower,
        "intersection_upper": intersection_upper,
        "intersection_width": intersection_upper - intersection_lower,
        "selector_signals": {} if selector_signals is None else dict(selector_signals),
    }
    return _ControllerConfiguredResult(
        lower=np.asarray(intersection_lower, dtype=float),
        upper=np.asarray(intersection_upper, dtype=float),
        attempt_results=list(successes),
        selected_result=selected,
        backend_name=selected.backend_name,
        machine_rigorous=bool(all(r.machine_rigorous for r in successes)),
        configuration_policy=configuration_policy,
        configuration_diagnostics=diagnostics,
        output_shape=tuple(getattr(selected, "output_shape", ())),
    )


def _run_auto_configuration_portfolio(
    model,
    intervals,
    *,
    order,
    backend,
    representation,
    range_policy,
    range_method,
    internal_range_method,
    final_range_method,
    branch_policy,
    configuration_policy,
):
    """Run the cheap Step-20a Phase-2 diagnostic-driven selector."""
    attempts = []
    successes = []
    shared_backend = backend if backend is not None else RigorousBinary64Backend()

    def run_candidate(candidate_representation, candidate_order, reason):
        started = time.perf_counter()
        try:
            result = _run_one_configuration(
                model,
                intervals,
                order=candidate_order,
                backend=shared_backend,
                representation=candidate_representation,
                range_policy=range_policy,
                range_method=range_method,
                internal_range_method=internal_range_method,
                final_range_method=final_range_method,
                branch_policy=branch_policy,
            )
        except AutoIAError as exc:
            elapsed = time.perf_counter() - started
            attempts.append({
                "representation": candidate_representation,
                "order": int(candidate_order),
                "reason": reason,
                "success": False,
                "runtime_seconds": float(elapsed),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            return None, exc
        elapsed = time.perf_counter() - started
        success_index = len(successes)
        successes.append(result)
        attempts.append({
            "representation": candidate_representation,
            "order": int(candidate_order),
            "reason": reason,
            "success": True,
            "runtime_seconds": float(elapsed),
            "success_index": int(success_index),
            "lower": np.asarray(result.lower, dtype=float).copy(),
            "upper": np.asarray(result.upper, dtype=float).copy(),
            "width": np.asarray(result.width, dtype=float).copy(),
            "structure": _configuration_structural_summary(result),
        })
        return result, None

    baseline, baseline_error = run_candidate(representation, int(order), "baseline")
    selector_signals = {}
    if baseline is not None:
        selector_signals = _configuration_structural_summary(baseline)
        candidates = _automatic_escalation_candidates(
            baseline,
            configuration_policy,
            baseline_representation=representation,
            baseline_order=int(order),
        )
    else:
        # Without a baseline certificate there are no structural diagnostics.
        # Retry only proof/configuration families that can plausibly change a
        # certification outcome; unsupported/model/input/internal errors are
        # not representation-order tuning problems.
        if isinstance(baseline_error, (UnsupportedOperationError, ModelExecutionError, InvalidInputError, InternalAutoIAError, ResourceLimitError)):
            raise baseline_error
        candidates = []
        if configuration_policy.auto_try_higher_order:
            candidates.append((
                representation,
                int(order) + int(configuration_policy.auto_order_increment),
                "baseline-failed-higher-order-fallback",
            ))
        if configuration_policy.auto_try_generator and representation != "generator":
            candidates.append((
                "generator", int(order), "baseline-failed-generator-fallback"
            ))
        candidates = tuple(candidates[: max(0, configuration_policy.max_attempts - 1)])

    for candidate_representation, candidate_order, reason in candidates:
        if len(attempts) >= configuration_policy.max_attempts:
            break
        run_candidate(candidate_representation, candidate_order, reason)

    if not successes:
        if baseline_error is not None:
            raise baseline_error
        raise InternalAutoIAError("Automatic configuration selector produced no successful attempt.")

    return _finalize_configuration_attempts(
        successes,
        attempts,
        configuration_policy=configuration_policy,
        baseline_representation=representation,
        baseline_order=int(order),
        selector_signals=selector_signals,
    )


def _run_reference_configuration_portfolio(
    model,
    intervals,
    *,
    order,
    backend,
    representation,
    range_policy,
    range_method,
    internal_range_method,
    final_range_method,
    branch_policy,
    configuration_policy,
):
    """Run the Step-20a Phase-1 bounded reference configuration portfolio.

    Every successful attempt is independently rigorous.  The returned object is
    one actual successful AutoIA result selected only by enclosure dominance;
    componentwise intersections across attempts are recorded as diagnostics but
    are not substituted for the selected result's structural objects.
    """
    candidates = _configuration_candidates(
        configuration_policy,
        baseline_representation=representation,
        baseline_order=int(order),
    )
    attempts = []
    successes = []
    first_error = None

    # Reuse one audited backend across candidate attempts.  Candidate runs remain
    # mathematically independent enclosures but need not repeat environment setup.
    shared_backend = backend if backend is not None else RigorousBinary64Backend()

    for candidate_representation, candidate_order in candidates:
        started = time.perf_counter()
        try:
            result = _run_one_configuration(
                model,
                intervals,
                order=candidate_order,
                backend=shared_backend,
                representation=candidate_representation,
                range_policy=range_policy,
                range_method=range_method,
                internal_range_method=internal_range_method,
                final_range_method=final_range_method,
                branch_policy=branch_policy,
            )
        except AutoIAError as exc:
            elapsed = time.perf_counter() - started
            if first_error is None:
                first_error = exc
            attempts.append({
                "representation": candidate_representation,
                "order": int(candidate_order),
                "success": False,
                "runtime_seconds": float(elapsed),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            continue

        elapsed = time.perf_counter() - started
        success_index = len(successes)
        successes.append(result)
        attempts.append({
            "representation": candidate_representation,
            "order": int(candidate_order),
            "success": True,
            "runtime_seconds": float(elapsed),
            "success_index": int(success_index),
            "lower": np.asarray(result.lower, dtype=float).copy(),
            "upper": np.asarray(result.upper, dtype=float).copy(),
            "width": np.asarray(result.width, dtype=float).copy(),
            "structure": _configuration_structural_summary(result),
        })

    if not successes:
        if first_error is not None:
            raise first_error
        raise InternalAutoIAError("Automatic configuration portfolio produced no attempts.")

    return _finalize_configuration_attempts(
        successes,
        attempts,
        configuration_policy=configuration_policy,
        baseline_representation=representation,
        baseline_order=int(order),
        selector_signals={"reference_oracle": True},
    )


def autoia(
    model: Callable[[Any], Any],
    parameter_intervals: Sequence[Interval],
    *,
    order: int | None = None,
    backend: RigorousBinary64Backend | None = None,
    representation: str | None = None,
    range_policy: RangePolicy | None = None,
    range_method: str | None = None,
    internal_range_method: str | None = None,
    final_range_method: str | None = None,
    branch_policy: BranchPolicy | None = None,
    configuration_policy: ConfigurationPolicy | None = None,
) -> AutoIAResult | AutoIAPathwiseResult | AutoIAConfiguredResult:
    """Certify ``model`` over a Cartesian box of uncertain parameters.

    If this function returns normally, every output interval satisfies the
    documented inclusion contract.  If it cannot return such a certificate it
    raises a subclass of :class:`autoia.AutoIAError` and returns no rigorous
    result.

    Version-1 novice semantics
    --------------------------
    ``autoia(model, box)`` uses the bounded automatic configuration controller
    with a polynomial/order-4 baseline.  A user who explicitly supplies
    ``order=`` or ``representation=`` is treated as requesting a fixed expert
    configuration unless ``configuration_policy`` is also supplied explicitly.
    Thus historical/research runs remain reproducible while the zero-knob call
    is genuinely automatic.
    """
    order_explicit = order is not None
    representation_explicit = representation is not None
    resolved_order = 4 if order is None else order
    resolved_representation = "polynomial" if representation is None else representation
    if configuration_policy is None:
        if order_explicit or representation_explicit:
            configuration_policy = ConfigurationPolicy(mode="fixed")
        else:
            configuration_policy = ConfigurationPolicy(mode="auto")

    intervals = _validate_common(
        model,
        parameter_intervals,
        order=resolved_order,
        backend=backend,
        representation=resolved_representation,
        range_policy=range_policy,
        range_method=range_method,
        internal_range_method=internal_range_method,
        final_range_method=final_range_method,
    )
    if branch_policy is not None and not isinstance(branch_policy, _ControllerBranchPolicy):
        raise InvalidInputError("branch_policy must be an autoia.BranchPolicy.")
    if configuration_policy is not None and not isinstance(
        configuration_policy, _ControllerConfigurationPolicy
    ):
        raise InvalidInputError(
            "configuration_policy must be an autoia.ConfigurationPolicy."
        )

    if configuration_policy is None or configuration_policy.mode == "fixed":
        result = _run_one_configuration(
            model, intervals, order=int(resolved_order), backend=backend,
            representation=resolved_representation, range_policy=range_policy,
            range_method=range_method, internal_range_method=internal_range_method,
            final_range_method=final_range_method, branch_policy=branch_policy,
        )
        return attach_input_specification(result, intervals)

    if configuration_policy.mode == "auto":
        result = _run_auto_configuration_portfolio(
            model, intervals, order=int(resolved_order), backend=backend,
            representation=resolved_representation, range_policy=range_policy,
            range_method=range_method, internal_range_method=internal_range_method,
            final_range_method=final_range_method, branch_policy=branch_policy,
            configuration_policy=configuration_policy,
        )
        return attach_input_specification(result, intervals)

    result = _run_reference_configuration_portfolio(
        model, intervals, order=int(resolved_order), backend=backend,
        representation=resolved_representation, range_policy=range_policy,
        range_method=range_method, internal_range_method=internal_range_method,
        final_range_method=final_range_method, branch_policy=branch_policy,
        configuration_policy=configuration_policy,
    )
    return attach_input_specification(result, intervals)


def autoia_subdivide(
    model: Callable[[Any], Any],
    parameter_intervals: Sequence[Interval],
    split_counts: Sequence[int],
    *,
    order: int = 4,
    backend: RigorousBinary64Backend | None = None,
    representation: str = "polynomial",
    range_policy: RangePolicy | None = None,
    range_method: str | None = None,
    internal_range_method: str | None = None,
    final_range_method: str | None = None,
) -> PiecewiseResult:
    """Certify a prescribed Cartesian subdivision of the input box."""
    intervals = _validate_common(
        model,
        parameter_intervals,
        order=order,
        backend=backend,
        representation=representation,
        range_policy=range_policy,
        range_method=range_method,
        internal_range_method=internal_range_method,
        final_range_method=final_range_method,
    )
    if np.isscalar(split_counts):
        counts = [int(split_counts)] * len(intervals)
    else:
        try:
            counts = list(split_counts)
        except Exception as exc:
            raise InvalidInputError("split_counts must be an integer or iterable of integers.") from exc
    if len(counts) != len(intervals):
        raise InvalidInputError("split_counts must have one entry per parameter.")
    for i, count in enumerate(counts):
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or int(count) < 1:
            raise InvalidInputError(
                "Subdivision counts must be positive integers.",
                context={"index": i, "value": count},
            )
    result = _invoke_internal(
        _controller_subdivide,
        _guard_model(model),
        intervals,
        counts,
        order=int(order),
        range_method=range_method,
        backend=backend,
        representation=representation,
        internal_range_method=internal_range_method,
        final_range_method=final_range_method,
        range_policy=range_policy,
    )
    return attach_input_specification(result, intervals)


def autoia_adaptive_subdivide(
    model: Callable[[Any], Any],
    parameter_intervals: Sequence[Interval],
    *,
    order: int = 4,
    backend: RigorousBinary64Backend | None = None,
    representation: str = "polynomial",
    range_policy: RangePolicy | None = None,
    subdivision_policy: SubdivisionPolicy | None = None,
) -> AdaptiveSubdivisionResult:
    """Certify with Version-0.11 structure-aware adaptive subdivision."""
    intervals = _validate_common(
        model,
        parameter_intervals,
        order=order,
        backend=backend,
        representation=representation,
        range_policy=range_policy,
    )
    if subdivision_policy is not None and not isinstance(
        subdivision_policy, _ControllerSubdivisionPolicy
    ):
        raise InvalidInputError("subdivision_policy must be an autoia.SubdivisionPolicy.")
    result = _invoke_internal(
        _controller_adaptive_subdivide,
        _guard_model(model),
        intervals,
        order=int(order),
        backend=backend,
        representation=representation,
        range_policy=range_policy,
        subdivision_policy=subdivision_policy,
    )
    return attach_input_specification(result, intervals)


__all__ = ["autoia", "autoia_subdivide", "autoia_adaptive_subdivide"]
