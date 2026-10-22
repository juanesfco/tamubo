"""Prepared and repeated-query AutoIA execution.

The ordinary :func:`autoia.autoia` entry point remains the simplest one-box API.
This module adds a persistent session for workloads that certify the same
engineering program on many different interval boxes.

A prepared session separates static program preparation from dynamic box proof:
local NumPy/SciPy dispatch is cloned once, one audited trusted backend is reused,
and a bounded certificate cache can answer exact repeated queries or rigorously
restrict eligible polynomial certificates to interior subboxes without
re-executing the engineering model.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import copy
import math
import multiprocessing as mp
import os
import inspect
import pickle
import threading
from typing import Iterable, Iterator, Sequence

import numpy as np

from .api import _guard_model, _invoke_internal, _validate_common, _finalize_configuration_attempts
from .backends import RigorousBinary64Backend
from .controller import autoia as _controller_autoia
from .controller.execution import rigorous_center_and_halfwidth
from .controller.ranging import adaptive_final_range
from .enclosures import (
    ValidatedValue,
    PolynomialEnclosure,
    IntervalEnclosure,
    SharedGeneratorPolynomialEnclosure,
    RangePolicyContext,
)
from .exceptions import InvalidInputError, InternalAutoIAError
from .intervals import Interval, attach_input_specification
from .policies import RangePolicy, BranchPolicy, ConfigurationPolicy
from .results import AutoIAResult, AutoIAPathwiseResult, AutoIAConfiguredResult
from .semantic import prepare_dispatch_model
from .trusted import point_interval
from .polynomial.recenter import rigorous_recenter_point_polynomial
from .controller.configuration import (
    configuration_candidates, automatic_escalation_candidates, structural_summary,
)
from .exceptions import AutoIAError, UnsupportedOperationError
import time


def _box_key(box: Sequence[Interval]) -> tuple[tuple[float, float], ...]:
    return tuple((float(I.lo), float(I.hi)) for I in box)


def _box_contains(parent: Sequence[Interval], child: Sequence[Interval]) -> bool:
    return len(parent) == len(child) and all(
        float(P.lo) <= float(C.lo) and float(C.hi) <= float(P.hi)
        for P, C in zip(parent, child)
    )


def _box_log_volume(box: Sequence[Interval]) -> float:
    # Used only to choose among already-sound containing certificates.
    total = 0.0
    for I in box:
        w = float(I.hi) - float(I.lo)
        if w <= 0.0:
            total += -800.0
        else:
            total += math.log(w)
    return total


def _normalize_one_box(box, expected_dim: int | None = None) -> list[Interval]:
    if isinstance(box, np.ndarray) and box.ndim == 2 and box.shape[-1] == 2:
        vals = [Interval(float(lo), float(hi)) for lo, hi in box]
    else:
        try:
            raw = list(box)
        except Exception as exc:
            raise InvalidInputError(
                "Each box must be an iterable of autoia.Interval objects or an (m,2) endpoint array."
            ) from exc
        vals = []
        for item in raw:
            if isinstance(item, Interval):
                vals.append(item)
            else:
                try:
                    pair = list(item)
                except Exception as exc:
                    raise InvalidInputError(
                        "Each box entry must be an autoia.Interval or a length-2 endpoint pair."
                    ) from exc
                if len(pair) != 2:
                    raise InvalidInputError("Endpoint pairs must have length 2.")
                vals.append(Interval(float(pair[0]), float(pair[1])))
    if expected_dim is not None and len(vals) != int(expected_dim):
        raise InvalidInputError(
            "Every repeated-query box must have the same parameter dimension.",
            context={"expected": int(expected_dim), "received": len(vals)},
        )
    return vals


def _normalize_boxes(boxes, expected_dim: int | None = None) -> list[list[Interval]]:
    arr = np.asarray(boxes, dtype=object) if isinstance(boxes, np.ndarray) else None
    if arr is not None and arr.ndim == 3 and arr.shape[-1] == 2:
        return [_normalize_one_box(arr[i], expected_dim) for i in range(arr.shape[0])]
    try:
        raw = list(boxes)
    except Exception as exc:
        raise InvalidInputError("boxes must be an iterable of interval boxes.") from exc
    return [_normalize_one_box(box, expected_dim) for box in raw]

def _resolve_workers(workers, n_items: int | None = None) -> int:
    if isinstance(workers, str):
        if workers.lower() != "auto":
            raise InvalidInputError("workers must be a positive integer or 'auto'.")
        value = max(1, int(os.cpu_count() or 1))
        if n_items is not None and int(n_items) > 0:
            value = min(value, int(n_items))
        return value
    if isinstance(workers, bool) or int(workers) < 1:
        raise InvalidInputError("workers must be a positive integer or 'auto'.")
    # Preserve explicit user requests even for short finite batches; this keeps
    # validation/failure semantics deterministic (e.g. explicit process mode
    # still checks whether the model is spawn-pickleable).
    return int(workers)




# V1 safe-auto many-box scheduling policy.  The default high-level
# enclose_many(..., workers="auto", parallel="auto") route does not pay
# process-pool startup cost for tiny batches.  Four default-sized chunks is a
# deliberately conservative crossover; explicit workers=/parallel= always
# override this strategy choice.  This constant affects scheduling only, never
# certificate validity.
_AUTO_MIN_PARALLEL_CHUNKS = 4


def _resolve_safe_auto_schedule(workers, parallel, n_items: int, chunk_size: int, model):
    requested_workers = workers
    requested_parallel = str(parallel).lower()
    if not (isinstance(requested_workers, str) and requested_workers.lower() == "auto"
            and requested_parallel == "auto"):
        resolved_workers = _resolve_workers(workers, n_items)
        return resolved_workers, _resolve_parallel(parallel, resolved_workers, model), "explicit-or-partial-override"

    if n_items <= 0:
        return 1, "serial", "auto-empty-batch"
    n_chunks = int(math.ceil(float(n_items) / float(chunk_size)))
    if n_chunks < _AUTO_MIN_PARALLEL_CHUNKS:
        return 1, "serial", f"auto-small-batch:{n_chunks}-chunks"
    cpu_workers = max(1, int(os.cpu_count() or 1))
    resolved_workers = min(cpu_workers, n_chunks)
    resolved_parallel = _resolve_parallel("auto", resolved_workers, model)
    return resolved_workers, resolved_parallel, f"auto-parallel:{n_chunks}-chunks"


def _serialize_model_for_process(model):
    """Serialize a model for spawn workers, preferring optional cloudpickle."""
    try:
        import cloudpickle
        return "cloudpickle", cloudpickle.dumps(model)
    except Exception:
        try:
            return "pickle", pickle.dumps(model)
        except Exception as exc:
            raise InvalidInputError(
                "Process-parallel AutoIA requires a serializable model. Install optional "
                "cloudpickle for local/closure functions, or use parallel='thread'."
            ) from exc


def _deserialize_process_model(payload):
    kind, data = payload
    if kind == "cloudpickle":
        import cloudpickle
        return cloudpickle.loads(data)
    return pickle.loads(data)


def _model_process_friendly(model) -> bool:
    try:
        _serialize_model_for_process(model)
        return True
    except InvalidInputError:
        return False


def _resolve_parallel(parallel: str, workers: int, model) -> str:
    parallel = str(parallel).lower()
    if parallel not in {"auto", "serial", "thread", "process"}:
        raise InvalidInputError("parallel must be 'auto', 'serial', 'thread', or 'process'.")
    if workers <= 1:
        return "serial"
    if parallel == "auto":
        return "process" if _model_process_friendly(model) else "thread"
    return parallel



@dataclass(frozen=True)
class PreparedSessionStats:
    model_preparations: int
    certify_requests: int
    engineering_model_executions: int
    exact_cache_hits: int
    restricted_cache_hits: int
    cache_misses: int
    cache_entries: int


@dataclass(frozen=True)
class ManyBoxExecutionInfo:
    """Non-proof execution metadata for one many-box certification call.

    These fields describe *how* independent boxes were scheduled.  They are
    diagnostics only and are never consulted by the trusted enclosure kernel.
    ``machine_rigorous`` refers to the returned CPU certificates, not to any
    optional proposal accelerator used elsewhere.
    """

    n_boxes: int
    requested_workers: int | str
    resolved_workers: int
    requested_parallel: str
    resolved_parallel: str
    chunk_size: int
    elapsed_seconds: float
    model_serializer: str | None
    machine_rigorous: bool
    n_succeeded: int | None = None
    n_failed: int = 0
    proof_backend: str = "cpu-binary64"
    scheduling_note: str | None = None

    @property
    def used_processes(self) -> bool:
        return self.resolved_parallel == "process"

    @property
    def used_threads(self) -> bool:
        return self.resolved_parallel == "thread"

    @property
    def used_cloudpickle(self) -> bool:
        return self.model_serializer == "cloudpickle"


@dataclass(frozen=True)
class ManyBoxFailure:
    """Serializable description of one box that did not certify.

    ``on_error="report"`` uses this record so one ordinary fail-closed box does
    not discard successful siblings.  Internal AutoIA invariant failures are
    never converted to reports; they still raise immediately.
    """

    index: int
    error_type: str
    error_code: str | None
    message: str
    context: dict
    suggestions: tuple[str, ...]


@dataclass(frozen=True)
class ManyBoxCertificationReport:
    """Per-box outcome returned by ``certify_many(..., on_error="report")``."""

    index: int
    ok: bool
    result: AutoIAResult | AutoIAPathwiseResult | AutoIAConfiguredResult | None = None
    failure: ManyBoxFailure | None = None


@dataclass(frozen=True)
class ManyBoxBoundsItemReport:
    """Per-box streaming bound outcome for ``iter_bounds_many`` report mode."""

    index: int
    ok: bool
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    failure: ManyBoxFailure | None = None


def _validate_on_error(on_error: str) -> str:
    mode = str(on_error).lower()
    if mode not in {"raise", "report"}:
        raise InvalidInputError("on_error must be 'raise' or 'report'.")
    return mode


def _failure_record(index: int, exc: AutoIAError) -> ManyBoxFailure:
    return ManyBoxFailure(
        index=int(index),
        error_type=type(exc).__name__,
        error_code=getattr(exc, "code", None),
        message=str(exc),
        context=dict(getattr(exc, "context", {}) or {}),
        suggestions=tuple(getattr(exc, "suggestions", ()) or ()),
    )


def _certify_one_with_policy(session, index, box, *, use_cache, allow_restriction, on_error):
    try:
        result = session.certify(
            box, use_cache=use_cache, allow_restriction=allow_restriction
        )
    except InternalAutoIAError:
        # Internal invariant failures are software defects, not ordinary
        # branch-and-bound refinement signals.  Never hide them in a batch.
        raise
    except AutoIAError as exc:
        if on_error == "raise":
            raise
        return ManyBoxCertificationReport(
            index=int(index), ok=False, failure=_failure_record(index, exc)
        )
    if on_error == "report":
        return ManyBoxCertificationReport(index=int(index), ok=True, result=result)
    return result


def _stack_many_bounds(results, on_error: str):
    """Convert raw or per-box certification outcomes to compact bound arrays."""
    if on_error == "raise":
        if not results:
            return np.empty((0, 0)), np.empty((0, 0))
        nout = len(results[0].lower)
        if any(len(r.lower) != nout for r in results):
            raise InvalidInputError(
                "All successful boxes must produce the same number of outputs."
            )
        return (
            np.vstack([np.asarray(r.lower, dtype=float) for r in results]),
            np.vstack([np.asarray(r.upper, dtype=float) for r in results]),
        )

    reports = list(results)
    ok = np.asarray([bool(r.ok) for r in reports], dtype=bool)
    successes = [r.result for r in reports if r.ok]
    if not successes:
        return np.empty((len(reports), 0)), np.empty((len(reports), 0)), ok
    nout = len(successes[0].lower)
    if any(len(r.lower) != nout for r in successes):
        raise InvalidInputError(
            "All successful boxes must produce the same number of outputs."
        )
    lower = np.full((len(reports), nout), np.nan, dtype=float)
    upper = np.full((len(reports), nout), np.nan, dtype=float)
    for i, item in enumerate(reports):
        if item.ok:
            lower[i, :] = np.asarray(item.result.lower, dtype=float)
            upper[i, :] = np.asarray(item.result.upper, dtype=float)
    return lower, upper, ok


@dataclass
class _CacheEntry:
    box: tuple[Interval, ...]
    result: AutoIAResult | AutoIAPathwiseResult
    centers: tuple[float, ...]
    halfwidths: tuple[float, ...]


class CertificateCache:
    """Bounded LRU cache of certified boxes for one prepared model/configuration.

    Exact lookup is O(1).  Containment lookup scans the bounded cache and picks
    the smallest containing eligible certificate.  The default cache is small
    deliberately; a future spatial index can replace this policy without
    changing :class:`PreparedAutoIA`'s public API.
    """

    def __init__(self, max_entries: int = 256):
        if isinstance(max_entries, bool) or int(max_entries) < 0:
            raise InvalidInputError("cache_max_entries must be a nonnegative integer.")
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[tuple, _CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

    def __len__(self):
        with self._lock:
            return len(self._entries)

    def clear(self):
        with self._lock:
            self._entries.clear()

    def get_exact(self, box):
        key = _box_key(box)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    def get_smallest_container(self, box):
        with self._lock:
            candidates = [
                (key, entry) for key, entry in self._entries.items()
                if _box_contains(entry.box, box)
            ]
            if not candidates:
                return None
            key, entry = min(candidates, key=lambda kv: _box_log_volume(kv[1].box))
            self._entries.move_to_end(key)
            return entry

    def put(self, box, result, centers, halfwidths):
        if self.max_entries == 0:
            return
        key = _box_key(box)
        entry = _CacheEntry(
            tuple(box), result, tuple(float(x) for x in centers),
            tuple(float(x) for x in halfwidths),
        )
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)


def _physical_symmetric_envelope(center: float, halfwidth: float, backend) -> Interval:
    return backend.add(point_interval(float(center)), Interval(-float(halfwidth), float(halfwidth)))


def _restrict_validated_value(value: ValidatedValue, shift, shift_intervals, new_h, backend):
    rep = value.rep
    name = rep.representation_name
    if name == "polynomial":
        coeff, transform_error = rigorous_recenter_point_polynomial(
            rep.coeff, shift, new_h, backend, shift_intervals=shift_intervals
        )
        rem = backend.add(rep.remainder, transform_error)
        return ValidatedValue(PolynomialEnclosure(
            coeff, rem, new_h, rep.order, backend, rep.range_method
        ), _diag_approx_mask=getattr(value, "_diag_approx_mask", 0),
           _diag_add_source=False,
           _diag_recombined=getattr(value, "_diag_recombined", False))
    if name == "generator-polynomial":
        coeff, transform_error = rigorous_recenter_point_polynomial(
            rep.coeff, shift, new_h, backend, shift_intervals=shift_intervals
        )
        rem = backend.add(rep.remainder, transform_error)
        return ValidatedValue(SharedGeneratorPolynomialEnclosure(
            coeff, rem, new_h, rep.order, backend, rep.range_method,
            generators=dict(rep.generators), lift_remainder=True,
        ), _diag_approx_mask=getattr(value, "_diag_approx_mask", 0),
           _diag_add_source=False,
           _diag_recombined=getattr(value, "_diag_recombined", False))
    if name == "interval":
        return ValidatedValue(IntervalEnclosure(
            rep.interval, new_h, rep.order, backend, rep.range_method
        ), _diag_approx_mask=getattr(value, "_diag_approx_mask", 0),
           _diag_add_source=False,
           _diag_recombined=getattr(value, "_diag_recombined", False))
    return None


def _restrict_cached_result(entry: _CacheEntry, query_box, range_policy, backend):
    """Rigorously restrict an eligible branchless cached result to ``query_box``.

    The existing certificate was proved over ``entry.box``.  We recenter its
    stored output polynomial only if the new *symmetric* centered domain used by
    AutoIA is itself contained in the certified parent box.  This check avoids
    the known one-ulp boundary enlargement issue.  Otherwise the caller simply
    performs a fresh certification.
    """
    parent_result = entry.result
    if not isinstance(parent_result, AutoIAResult) or isinstance(parent_result, AutoIAPathwiseResult):
        return None

    interval_only = all(
        getattr(v, "representation_name", None) == "interval"
        for v in parent_result.objects
    )
    centers = np.empty(len(query_box), dtype=float)
    new_h = np.empty(len(query_box), dtype=float)
    for i, I in enumerate(query_box):
        centers[i], new_h[i] = rigorous_center_and_halfwidth(I, backend)
        if not interval_only:
            envelope = _physical_symmetric_envelope(centers[i], new_h[i], backend)
            P = entry.box[i]
            if envelope.lo < P.lo or envelope.hi > P.hi:
                return None

    old_centers = np.asarray(entry.centers, dtype=float)
    shift = centers - old_centers
    shift_intervals = [
        backend.sub(point_interval(float(c_new)), point_interval(float(c_old)))
        for c_new, c_old in zip(centers, old_centers)
    ]
    objects = []
    lo = []
    hi = []
    final_meta = []
    policy_context = RangePolicyContext(range_policy)
    with policy_context:
        for old in parent_result.objects:
            new_value = _restrict_validated_value(
                old, shift, shift_intervals, new_h, backend
            )
            if new_value is None:
                return None
            r, meta = adaptive_final_range(new_value, range_policy)
            objects.append(new_value)
            lo.append(r.lo)
            hi.append(r.hi)
            final_meta.append(meta)

    out = AutoIAResult(
        np.asarray(lo), np.asarray(hi), objects,
        backend.name, backend.machine_rigorous,
        list(parent_result.certificates),
        parent_result.requested_representation,
        range_policy, final_meta, list(policy_context.events),
        parent_result.output_shape,
    )
    out.approximation_diagnostics = dict(getattr(parent_result, "approximation_diagnostics", {}))
    out.cache_info = {
        "kind": "restricted-certificate",
        "parent_box": _box_key(entry.box),
        "query_box": _box_key(query_box),
        "model_reexecuted": False,
    }
    return out


# ---------------------------------------------------------------------------
# Optional process workers.  A worker prepares the engineering function once,
# then handles all chunks assigned to that process with its own trusted backend
# and local certificate cache.
# ---------------------------------------------------------------------------
_PROCESS_SESSION = None


def _process_initializer(model_payload, config):
    global _PROCESS_SESSION
    model = _deserialize_process_model(model_payload)
    backend = RigorousBinary64Backend(
        reduction_method=config["reduction_method"],
        vectorized_reduction_min_terms=config["vectorized_reduction_min_terms"],
    )
    policy = config.get("configuration_policy")
    cls = PreparedConfiguredAutoIA if policy is not None else PreparedAutoIA
    kwargs = dict(
        order=config["order"],
        backend=backend,
        representation=config["representation"],
        range_policy=config["range_policy"],
        branch_policy=config["branch_policy"],
        cache_max_entries=config["cache_max_entries"],
    )
    if policy is not None:
        kwargs["configuration_policy"] = policy
    _PROCESS_SESSION = cls(model, **kwargs)


def _process_certify_chunk(payload):
    indexed_boxes, use_cache, allow_restriction, on_error = payload
    out = []
    for idx, box in indexed_boxes:
        item = _certify_one_with_policy(
            _PROCESS_SESSION, idx, box, use_cache=use_cache,
            allow_restriction=allow_restriction, on_error=on_error,
        )
        out.append((idx, item))
    return out


class PreparedManyBoxExecutor:
    """Persistent executor for repeated dynamic batches of proof boxes.

    Unlike a one-shot process-parallel ``certify_many`` call, this explicit
    context manager keeps workers alive across successive calls.  Each process
    prepares the model once and reuses its own audited CPU proof backend/cache.
    This is scheduling only; every returned box is independently certified.
    """

    def __init__(self, session, *, workers="auto", parallel="auto",
                 chunk_size=1, use_cache=True, allow_restriction=True,
                 on_error="raise"):
        if isinstance(chunk_size, bool) or int(chunk_size) < 1:
            raise InvalidInputError("chunk_size must be a positive integer.")
        self.session = session
        self.requested_workers = workers
        self.requested_parallel = str(parallel).lower()
        self.chunk_size = int(chunk_size)
        self.use_cache = bool(use_cache)
        self.allow_restriction = bool(allow_restriction)
        self.on_error = _validate_on_error(on_error)
        self.workers = _resolve_workers(workers, None)
        self.parallel = _resolve_parallel(self.requested_parallel, self.workers, session.model)
        self._executor = None
        self._serializer = None
        self._closed = False
        self._started = False
        self._startup_seconds = 0.0
        self._calls = 0
        self._boxes = 0
        self._last_info = None

    @property
    def startup_seconds(self): return float(self._startup_seconds)
    @property
    def calls(self): return int(self._calls)
    @property
    def boxes_certified(self): return int(self._boxes)
    @property
    def last_manybox_execution(self): return self._last_info

    def _ensure_started(self):
        if self._closed:
            raise RuntimeError("PreparedManyBoxExecutor is closed.")
        if self._started:
            return
        t0 = time.perf_counter()
        if self.parallel == "thread":
            self._executor = ThreadPoolExecutor(max_workers=self.workers)
        elif self.parallel == "process":
            model_payload = _serialize_model_for_process(self.session.model)
            self._serializer = str(model_payload[0])
            self._executor = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=mp.get_context("spawn"),
                initializer=_process_initializer,
                initargs=(model_payload, self.session._process_config()),
            )
        self._startup_seconds = float(time.perf_counter() - t0)
        self._started = True

    def __enter__(self):
        self._ensure_started(); return self

    def close(self):
        if self._closed: return
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._executor = None
        self._closed = True

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(); return False

    def _expected_dimension(self):
        if hasattr(self.session, "_dimension"):
            return self.session._dimension
        baseline = getattr(self.session, "baseline", None)
        return getattr(baseline, "_dimension", None)

    def _record_dimension(self, dim):
        if hasattr(self.session, "_dimension") and self.session._dimension is None:
            self.session._dimension = int(dim)
        baseline = getattr(self.session, "baseline", None)
        if baseline is not None and getattr(baseline, "_dimension", None) is None:
            baseline._dimension = int(dim)

    def certify_many(self, boxes, *, on_error=None):
        self._ensure_started()
        t0 = time.perf_counter()
        mode = self.on_error if on_error is None else _validate_on_error(on_error)
        normalized = _normalize_boxes(boxes, self._expected_dimension())
        if normalized:
            self._record_dimension(len(normalized[0]))
        indexed = list(enumerate(normalized))
        if self.parallel == "serial":
            results = [
                _certify_one_with_policy(
                    self.session, idx, b, use_cache=self.use_cache,
                    allow_restriction=self.allow_restriction, on_error=mode,
                )
                for idx, b in indexed
            ]
        elif self.parallel == "thread":
            def one(pair):
                idx, b = pair
                return _certify_one_with_policy(
                    self.session, idx, b, use_cache=self.use_cache,
                    allow_restriction=self.allow_restriction, on_error=mode,
                )
            results = list(self._executor.map(one, indexed))
        else:
            chunks = [indexed[i:i+self.chunk_size] for i in range(0, len(indexed), self.chunk_size)]
            payloads = [
                (chunk, self.use_cache, self.allow_restriction, mode)
                for chunk in chunks
            ]
            collected = []
            for part in self._executor.map(_process_certify_chunk, payloads):
                collected.extend(part)
            collected.sort(key=lambda t: t[0])
            results = [r for _, r in collected]

        if mode == "report":
            successes = [r.result for r in results if r.ok]
            n_failed = sum(not r.ok for r in results)
        else:
            successes = results
            n_failed = 0
        info = ManyBoxExecutionInfo(
            n_boxes=len(normalized), requested_workers=self.requested_workers,
            resolved_workers=int(self.workers), requested_parallel=self.requested_parallel,
            resolved_parallel=str(self.parallel), chunk_size=self.chunk_size,
            elapsed_seconds=float(time.perf_counter()-t0), model_serializer=self._serializer,
            machine_rigorous=bool(all(getattr(r, "machine_rigorous", False) for r in successes)),
            scheduling_note="persistent-manybox-executor",
            n_succeeded=len(successes), n_failed=int(n_failed),
        )
        self._calls += 1; self._boxes += len(normalized); self._last_info = info
        if hasattr(self.session, "_lock") and hasattr(self.session, "_last_manybox_execution"):
            with self.session._lock:
                self.session._last_manybox_execution = info
        return results

    def enclose_many(self, boxes, *, return_info=False, on_error=None):
        results = self.certify_many(boxes, on_error=on_error)
        return (results, self._last_info) if return_info else results

    def enclose_many_bounds(self, boxes, *, return_info=False, on_error=None):
        mode = self.on_error if on_error is None else _validate_on_error(on_error)
        results = self.certify_many(boxes, on_error=mode)
        packed = _stack_many_bounds(results, mode)
        if return_info:
            return (*packed, self._last_info)
        return packed


class PreparedAutoIA:
    """Prepared repeated-query session for one engineering program/configuration."""

    def __init__(
        self,
        model,
        *,
        order: int = 4,
        backend: RigorousBinary64Backend | None = None,
        representation: str = "polynomial",
        range_policy: RangePolicy | None = None,
        range_method: str | None = None,
        internal_range_method: str | None = None,
        final_range_method: str | None = None,
        branch_policy: BranchPolicy | None = None,
        cache_max_entries: int = 256,
    ):
        if not callable(model):
            raise InvalidInputError("model must be callable.")
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or int(order) < 0:
            raise InvalidInputError("order must be a nonnegative integer.")
        if representation not in {"polynomial", "interval", "generator"}:
            raise InvalidInputError("representation must be 'polynomial', 'interval', or 'generator'.")
        if backend is not None and not isinstance(backend, RigorousBinary64Backend):
            raise InvalidInputError("backend must be RigorousBinary64Backend or None.")
        if range_policy is not None and any(
            v is not None for v in (range_method, internal_range_method, final_range_method)
        ):
            raise InvalidInputError(
                "Pass either range_policy or the range_method/internal/final options, not both."
            )
        from .controller.execution import resolve_range_policy
        self.range_policy = resolve_range_policy(
            range_method, internal_range_method, final_range_method, range_policy
        )
        if branch_policy is not None and not isinstance(branch_policy, BranchPolicy):
            raise InvalidInputError("branch_policy must be an autoia.BranchPolicy.")

        self.model = model
        self.order = int(order)
        self.backend = backend if backend is not None else RigorousBinary64Backend()
        self.representation = str(representation)
        self.branch_policy = BranchPolicy() if branch_policy is None else branch_policy
        self.cache = CertificateCache(cache_max_entries)

        self._guarded_model = _guard_model(model)
        self._prepared_dispatch = prepare_dispatch_model(self._guarded_model)
        self._dimension = None
        self._lock = threading.RLock()
        self._certify_requests = 0
        self._model_executions = 0
        self._exact_cache_hits = 0
        self._restricted_cache_hits = 0
        self._cache_misses = 0
        self._last_manybox_execution = None

    @property
    def stats(self) -> PreparedSessionStats:
        with self._lock:
            return PreparedSessionStats(
                model_preparations=1,
                certify_requests=self._certify_requests,
                engineering_model_executions=self._model_executions,
                exact_cache_hits=self._exact_cache_hits,
                restricted_cache_hits=self._restricted_cache_hits,
                cache_misses=self._cache_misses,
                cache_entries=len(self.cache),
            )

    @property
    def last_manybox_execution(self) -> ManyBoxExecutionInfo | None:
        """Execution metadata from the most recent finite many-box call."""
        with self._lock:
            return self._last_manybox_execution

    def clear_cache(self):
        self.cache.clear()

    def _validated_box(self, box):
        intervals = _normalize_one_box(box, self._dimension)
        # Reuse the stable public validation semantics without preparing/executing.
        intervals = _validate_common(
            self.model, intervals, order=self.order, backend=self.backend,
            representation=self.representation, range_policy=self.range_policy,
        )
        if self._dimension is None:
            self._dimension = len(intervals)
        return intervals

    def certify(self, parameter_intervals, *, use_cache: bool = True, allow_restriction: bool = True):
        """Certify one box using the prepared program and persistent cache."""
        box = self._validated_box(parameter_intervals)
        with self._lock:
            self._certify_requests += 1

        if use_cache:
            exact = self.cache.get_exact(box)
            if exact is not None:
                with self._lock:
                    self._exact_cache_hits += 1
                # Return a shallow result copy before attaching per-request cache
                # metadata.  The cached certificate itself is shared, but a later
                # exact lookup must not retroactively mutate ``cache_info`` on a
                # result object already handed to the user.
                result = copy.copy(exact.result)
                result.cache_info = {
                    "kind": "exact-certificate",
                    "query_box": _box_key(box),
                    "model_reexecuted": False,
                }
                return attach_input_specification(result, box)

            if allow_restriction:
                parent = self.cache.get_smallest_container(box)
                if parent is not None:
                    restricted = _restrict_cached_result(
                        parent, box, self.range_policy, self.backend
                    )
                    if restricted is not None:
                        centers = []
                        h = []
                        for I in box:
                            c, hi = rigorous_center_and_halfwidth(I, self.backend)
                            centers.append(c); h.append(hi)
                        self.cache.put(box, restricted, centers, h)
                        with self._lock:
                            self._restricted_cache_hits += 1
                        return attach_input_specification(restricted, box)

        with self._lock:
            self._cache_misses += 1
            self._model_executions += 1
        result = _invoke_internal(
            _controller_autoia,
            self._guarded_model,
            box,
            order=self.order,
            backend=self.backend,
            representation=self.representation,
            range_policy=self.range_policy,
            branch_policy=self.branch_policy,
            prepared_dispatch=self._prepared_dispatch,
        )
        result.cache_info = {
            "kind": "fresh-certificate",
            "query_box": _box_key(box),
            "model_reexecuted": True,
        }
        attach_input_specification(result, box)
        centers = []
        h = []
        for I in box:
            c, hi = rigorous_center_and_halfwidth(I, self.backend)
            centers.append(c); h.append(hi)
        if use_cache:
            self.cache.put(box, result, centers, h)
        return result

    def _process_config(self):
        return {
            "order": self.order,
            "representation": self.representation,
            "range_policy": self.range_policy,
            "branch_policy": self.branch_policy,
            "cache_max_entries": self.cache.max_entries,
            "reduction_method": self.backend.reduction_method,
            "vectorized_reduction_min_terms": self.backend.vectorized_reduction_min_terms,
        }

    def certify_many(
        self,
        boxes,
        *,
        workers: int | str = 1,
        parallel: str = "serial",
        chunk_size: int = 64,
        use_cache: bool = True,
        allow_restriction: bool = True,
        on_error: str = "raise",
    ):
        """Certify many boxes, preserving input order.

        ``parallel='thread'`` shares this prepared session and cache.
        ``parallel='process'`` gives each worker one independently prepared
        session and therefore exploits multiple CPU cores for Python-heavy
        proof work.  Optional ``cloudpickle`` is used when available so local
        and closure functions can participate in spawn-based process pools.

        Parallel scheduling is non-proof infrastructure: every returned result
        is independently produced by the same trusted CPU binary64 backend.
        """
        t0 = time.perf_counter()
        requested_workers = workers
        requested_parallel = str(parallel).lower()
        on_error = _validate_on_error(on_error)
        normalized = _normalize_boxes(boxes, self._dimension)
        if normalized and self._dimension is None:
            self._dimension = len(normalized[0])
        if isinstance(chunk_size, bool) or int(chunk_size) < 1:
            raise InvalidInputError("chunk_size must be a positive integer.")
        chunk_size = int(chunk_size)
        workers, parallel, scheduling_note = _resolve_safe_auto_schedule(
            workers, parallel, len(normalized), chunk_size, self.model
        )
        serializer = None

        indexed = list(enumerate(normalized))
        if parallel == "serial":
            results = [
                _certify_one_with_policy(
                    self, idx, b, use_cache=use_cache,
                    allow_restriction=allow_restriction, on_error=on_error,
                )
                for idx, b in indexed
            ]
        elif parallel == "thread":
            def run_chunk(chunk):
                return [
                    _certify_one_with_policy(
                        self, idx, b, use_cache=use_cache,
                        allow_restriction=allow_restriction, on_error=on_error,
                    )
                    for idx, b in chunk
                ]
            chunks = [indexed[i:i+chunk_size] for i in range(0, len(indexed), chunk_size)]
            results = []
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for chunk_results in ex.map(run_chunk, chunks):
                    results.extend(chunk_results)
        else:
            # Process mode prepares the model once per worker process.
            model_payload = _serialize_model_for_process(self.model)
            serializer = str(model_payload[0])
            chunks = [indexed[i:i+chunk_size] for i in range(0, len(indexed), chunk_size)]
            collected = []
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
                initializer=_process_initializer,
                initargs=(model_payload, self._process_config()),
            ) as ex:
                payloads = [
                    (chunk, bool(use_cache), bool(allow_restriction), on_error)
                    for chunk in chunks
                ]
                for part in ex.map(_process_certify_chunk, payloads):
                    collected.extend(part)
            collected.sort(key=lambda item: item[0])
            results = [r for _, r in collected]

        info = ManyBoxExecutionInfo(
            n_boxes=len(normalized),
            requested_workers=requested_workers,
            resolved_workers=int(workers),
            requested_parallel=requested_parallel,
            resolved_parallel=str(parallel),
            chunk_size=int(chunk_size),
            elapsed_seconds=float(time.perf_counter() - t0),
            model_serializer=serializer,
            machine_rigorous=bool(all(
                getattr((r.result if isinstance(r, ManyBoxCertificationReport) else r),
                        "machine_rigorous", False)
                for r in results if not isinstance(r, ManyBoxCertificationReport) or r.ok
            )),
            scheduling_note=scheduling_note,
            n_succeeded=sum(
                1 for r in results
                if not isinstance(r, ManyBoxCertificationReport) or r.ok
            ),
            n_failed=sum(
                1 for r in results
                if isinstance(r, ManyBoxCertificationReport) and not r.ok
            ),
        )
        with self._lock:
            self._last_manybox_execution = info
        return results

    def certify_many_bounds(self, boxes, **kwargs):
        """Return stacked bounds; report mode also returns a per-box ``ok`` mask.

        Failed rows are filled with NaN and must be ignored unless ``ok[i]`` is
        true.  Use ``certify_many(..., on_error="report")`` when the detailed
        per-box failure code/message is also needed.
        """
        on_error = _validate_on_error(kwargs.get("on_error", "raise"))
        results = self.certify_many(boxes, **kwargs)
        return _stack_many_bounds(results, on_error)

    # V1 naming aliases: "enclose" matches the conceptual AutoIA operation,
    # while the historical ``certify_many`` spelling remains supported.
    def enclose_many(self, boxes, *, workers="auto", parallel="auto", return_info=False, **kwargs):
        results = self.certify_many(boxes, workers=workers, parallel=parallel, **kwargs)
        return (results, self.last_manybox_execution) if return_info else results

    def enclose_many_bounds(self, boxes, *, workers="auto", parallel="auto", return_info=False, **kwargs):
        packed = self.certify_many_bounds(boxes, workers=workers, parallel=parallel, **kwargs)
        if return_info:
            return (*packed, self.last_manybox_execution)
        return packed

    def manybox_executor(self, *, workers="auto", parallel="auto", chunk_size=1,
                         use_cache=True, allow_restriction=True, on_error="raise"):
        """Return a persistent executor for successive box batches."""
        return PreparedManyBoxExecutor(
            self, workers=workers, parallel=parallel, chunk_size=chunk_size,
            use_cache=use_cache, allow_restriction=allow_restriction,
            on_error=on_error,
        )

    def iter_bounds_many(self, boxes: Iterable, **kwargs) -> Iterator:
        """Stream bounds; report mode yields :class:`ManyBoxBoundsItemReport`."""
        on_error = _validate_on_error(kwargs.get("on_error", "raise"))
        for item in self.iter_certify_many(boxes, **kwargs):
            if on_error == "raise":
                yield (
                    np.asarray(item.lower, dtype=float).copy(),
                    np.asarray(item.upper, dtype=float).copy(),
                )
            elif item.ok:
                yield ManyBoxBoundsItemReport(
                    index=item.index, ok=True,
                    lower=np.asarray(item.result.lower, dtype=float).copy(),
                    upper=np.asarray(item.result.upper, dtype=float).copy(),
                )
            else:
                yield ManyBoxBoundsItemReport(
                    index=item.index, ok=False, failure=item.failure
                )

    def iter_certify_many(
        self,
        boxes: Iterable,
        *,
        workers: int | str = 1,
        parallel: str = "serial",
        chunk_size: int = 64,
        use_cache: bool = True,
        allow_restriction: bool = True,
        on_error: str = "raise",
    ) -> Iterator:
        """Stream per-box results while keeping one executor alive.

        ``on_error="report"`` converts ordinary public AutoIA fail-closed
        exceptions into :class:`ManyBoxCertificationReport` records so later
        boxes are still processed.  ``InternalAutoIAError`` and unexpected
        exceptions still propagate immediately.
        """
        on_error = _validate_on_error(on_error)
        workers = _resolve_workers(workers, None)
        if isinstance(chunk_size, bool) or int(chunk_size) < 1:
            raise InvalidInputError("chunk_size must be a positive integer.")
        chunk_size = int(chunk_size)
        parallel = _resolve_parallel(parallel, workers, self.model)

        next_index = 0
        def chunks_from_stream():
            nonlocal next_index
            buf = []
            for raw in boxes:
                box = _normalize_one_box(raw, self._dimension)
                if self._dimension is None:
                    self._dimension = len(box)
                buf.append((next_index, box))
                next_index += 1
                if len(buf) >= chunk_size:
                    yield buf
                    buf = []
            if buf:
                yield buf

        if parallel == "serial":
            for chunk in chunks_from_stream():
                for idx, box in chunk:
                    yield _certify_one_with_policy(
                        self, idx, box, use_cache=use_cache,
                        allow_restriction=allow_restriction, on_error=on_error,
                    )
            return

        from collections import deque
        if parallel == "thread":
            def run_chunk(chunk):
                return [
                    _certify_one_with_policy(
                        self, idx, b, use_cache=use_cache,
                        allow_restriction=allow_restriction, on_error=on_error,
                    )
                    for idx, b in chunk
                ]
            with ThreadPoolExecutor(max_workers=workers) as ex:
                pending = deque()
                for chunk in chunks_from_stream():
                    pending.append(ex.submit(run_chunk, chunk))
                    if len(pending) >= workers:
                        for result in pending.popleft().result():
                            yield result
                while pending:
                    for result in pending.popleft().result():
                        yield result
            return

        model_payload = _serialize_model_for_process(self.model)
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_process_initializer,
            initargs=(model_payload, self._process_config()),
        ) as ex:
            pending = deque()
            for chunk in chunks_from_stream():
                payload = (chunk, bool(use_cache), bool(allow_restriction), on_error)
                pending.append(ex.submit(_process_certify_chunk, payload))
                if len(pending) >= workers:
                    part = pending.popleft().result()
                    part.sort(key=lambda item: item[0])
                    for _, result in part:
                        yield result
            while pending:
                part = pending.popleft().result()
                part.sort(key=lambda item: item[0])
                for _, result in part:
                    yield result

    def propose_centers_many(self, proposal_model, boxes, *, accelerator="auto", dtype=None):
        """Evaluate a Torch-compatible non-rigorous proposal at box centers.

        MPS/CUDA proposal values never become certificates.  Use ``certify_many``
        or ``enclose_many`` for the independent trusted CPU proof.
        """
        from .accelerators import proposal_box_centers
        normalized = _normalize_boxes(boxes, self._dimension)
        endpoint_boxes = [[(I.lo, I.hi) for I in b] for b in normalized]
        return proposal_box_centers(
            proposal_model, endpoint_boxes, device=accelerator, dtype=dtype
        )



@dataclass(frozen=True)
class PreparedConfiguredSessionStats:
    model_preparations: int
    certify_requests: int
    configuration_attempts: int
    escalated_requests: int
    engineering_model_executions: int
    exact_cache_hits: int
    restricted_cache_hits: int
    cache_misses: int
    cache_entries: int
    candidate_sessions: int


class PreparedConfiguredAutoIA:
    """Prepared repeated-query session with lazy automatic configuration retries.

    The baseline prepared session is created immediately.  Alternate order or
    generator sessions are prepared only if at least one queried box exposes a
    corresponding structural trigger, then reused for all later boxes.
    """

    def __init__(
        self,
        model,
        *,
        order: int = 4,
        backend: RigorousBinary64Backend | None = None,
        representation: str = "polynomial",
        range_policy: RangePolicy | None = None,
        range_method: str | None = None,
        internal_range_method: str | None = None,
        final_range_method: str | None = None,
        branch_policy: BranchPolicy | None = None,
        configuration_policy: ConfigurationPolicy | None = None,
        cache_max_entries: int = 256,
    ):
        if configuration_policy is None:
            configuration_policy = ConfigurationPolicy(mode="auto")
        if not isinstance(configuration_policy, ConfigurationPolicy):
            raise InvalidInputError("configuration_policy must be an autoia.ConfigurationPolicy.")
        if configuration_policy.mode == "fixed":
            raise InvalidInputError(
                "PreparedConfiguredAutoIA requires configuration_policy mode 'auto' or 'reference'."
            )
        self.model = model
        self.order = int(order)
        self.representation = str(representation)
        self.configuration_policy = configuration_policy
        self.backend = backend if backend is not None else RigorousBinary64Backend()
        self.cache_max_entries = int(cache_max_entries)
        self._lock = threading.RLock()
        self._certify_requests = 0
        self._configuration_attempts = 0
        self._escalated_requests = 0
        self._last_manybox_execution = None

        self.baseline = PreparedAutoIA(
            model,
            order=self.order,
            backend=self.backend,
            representation=self.representation,
            range_policy=range_policy,
            range_method=range_method,
            internal_range_method=internal_range_method,
            final_range_method=final_range_method,
            branch_policy=branch_policy,
            cache_max_entries=self.cache_max_entries,
        )
        self.range_policy = self.baseline.range_policy
        self.branch_policy = self.baseline.branch_policy
        self._sessions = {(self.representation, self.order): self.baseline}

    @property
    def last_manybox_execution(self) -> ManyBoxExecutionInfo | None:
        with self._lock:
            return self._last_manybox_execution

    def _session_for(self, representation, order):
        key = (str(representation), int(order))
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                session = PreparedAutoIA(
                    self.model,
                    order=int(order),
                    backend=self.backend,
                    representation=str(representation),
                    range_policy=self.range_policy,
                    branch_policy=self.branch_policy,
                    cache_max_entries=self.cache_max_entries,
                )
                self._sessions[key] = session
            return session

    @property
    def stats(self) -> PreparedConfiguredSessionStats:
        with self._lock:
            sessions = list(self._sessions.values())
            requests = self._certify_requests
            attempts = self._configuration_attempts
            escalated = self._escalated_requests
        stats = [s.stats for s in sessions]
        return PreparedConfiguredSessionStats(
            model_preparations=int(sum(s.model_preparations for s in stats)),
            certify_requests=int(requests),
            configuration_attempts=int(attempts),
            escalated_requests=int(escalated),
            engineering_model_executions=int(sum(s.engineering_model_executions for s in stats)),
            exact_cache_hits=int(sum(s.exact_cache_hits for s in stats)),
            restricted_cache_hits=int(sum(s.restricted_cache_hits for s in stats)),
            cache_misses=int(sum(s.cache_misses for s in stats)),
            cache_entries=int(sum(s.cache_entries for s in stats)),
            candidate_sessions=int(len(sessions)),
        )

    def clear_cache(self):
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.clear_cache()

    def certify(self, parameter_intervals, *, use_cache: bool = True, allow_restriction: bool = True):
        with self._lock:
            self._certify_requests += 1

        attempts = []
        successes = []
        baseline_error = None
        box = _normalize_one_box(parameter_intervals, self.baseline._dimension)

        def run(rep, order, reason):
            session = self._session_for(rep, order)
            t0 = time.perf_counter()
            try:
                result = session.certify(
                    box, use_cache=use_cache, allow_restriction=allow_restriction
                )
            except AutoIAError as exc:
                attempts.append({
                    "representation": str(rep), "order": int(order), "reason": reason,
                    "success": False, "runtime_seconds": float(time.perf_counter()-t0),
                    "error_type": type(exc).__name__, "error_message": str(exc),
                })
                with self._lock:
                    self._configuration_attempts += 1
                return None, exc
            success_index = len(successes)
            successes.append(result)
            attempts.append({
                "representation": str(rep), "order": int(order), "reason": reason,
                "success": True, "runtime_seconds": float(time.perf_counter()-t0),
                "success_index": int(success_index),
                "lower": np.asarray(result.lower, dtype=float).copy(),
                "upper": np.asarray(result.upper, dtype=float).copy(),
                "width": np.asarray(result.width, dtype=float).copy(),
                "structure": structural_summary(result),
                "cache_info": dict(getattr(result, "cache_info", {})),
            })
            with self._lock:
                self._configuration_attempts += 1
            return result, None

        baseline, baseline_error = run(self.representation, self.order, "baseline")
        selector_signals = {}
        if baseline is not None:
            selector_signals = structural_summary(baseline)
            if self.configuration_policy.mode == "reference":
                raw = configuration_candidates(
                    self.configuration_policy,
                    baseline_representation=self.representation,
                    baseline_order=self.order,
                )[1:]
                candidates = tuple((rep, order, "reference-portfolio") for rep, order in raw)
            else:
                candidates = automatic_escalation_candidates(
                    baseline,
                    self.configuration_policy,
                    baseline_representation=self.representation,
                    baseline_order=self.order,
                )
        else:
            candidates = []
            if self.configuration_policy.auto_try_higher_order:
                candidates.append((
                    self.representation,
                    self.order + self.configuration_policy.auto_order_increment,
                    "baseline-failed-higher-order-fallback",
                ))
            if self.configuration_policy.auto_try_generator and self.representation != "generator":
                candidates.append(("generator", self.order, "baseline-failed-generator-fallback"))

        for rep, order, reason in candidates:
            if len(attempts) >= self.configuration_policy.max_attempts:
                break
            run(rep, order, reason)

        if len(attempts) > 1:
            with self._lock:
                self._escalated_requests += 1
        if not successes:
            if baseline_error is not None:
                raise baseline_error
            raise InvalidInputError("Prepared automatic configuration produced no successful attempt.")

        result = _finalize_configuration_attempts(
            successes,
            attempts,
            configuration_policy=self.configuration_policy,
            baseline_representation=self.representation,
            baseline_order=self.order,
            selector_signals=selector_signals,
        )
        result.cache_info = {
            "kind": "prepared-configuration-intersection",
            "query_box": _box_key(box),
            "attempts": len(attempts),
            "model_reexecuted": any(
                bool(a.get("cache_info", {}).get("model_reexecuted", False))
                for a in attempts if a.get("success")
            ),
        }
        return attach_input_specification(result, box)

    def _process_config(self):
        return {
            "order": self.order,
            "representation": self.representation,
            "range_policy": self.range_policy,
            "branch_policy": self.branch_policy,
            "configuration_policy": self.configuration_policy,
            "cache_max_entries": self.cache_max_entries,
            "reduction_method": self.backend.reduction_method,
            "vectorized_reduction_min_terms": self.backend.vectorized_reduction_min_terms,
        }

    def certify_many(
        self, boxes, *, workers: int | str = 1, parallel: str = "serial",
        chunk_size: int = 64, use_cache: bool = True, allow_restriction: bool = True,
        on_error: str = "raise",
    ):
        t0 = time.perf_counter()
        requested_workers = workers
        requested_parallel = str(parallel).lower()
        on_error = _validate_on_error(on_error)
        normalized = _normalize_boxes(boxes, self.baseline._dimension)
        if isinstance(chunk_size, bool) or int(chunk_size) < 1:
            raise InvalidInputError("chunk_size must be a positive integer.")
        chunk_size = int(chunk_size)
        workers, parallel, scheduling_note = _resolve_safe_auto_schedule(
            workers, parallel, len(normalized), chunk_size, self.model
        )
        serializer = None
        indexed = list(enumerate(normalized))
        if parallel == "serial":
            results = [
                _certify_one_with_policy(
                    self, idx, b, use_cache=use_cache,
                    allow_restriction=allow_restriction, on_error=on_error,
                )
                for idx, b in indexed
            ]
        elif parallel == "thread":
            def one(pair):
                idx, b = pair
                return _certify_one_with_policy(
                    self, idx, b, use_cache=use_cache,
                    allow_restriction=allow_restriction, on_error=on_error,
                )
            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(one, indexed))
        else:
            model_payload = _serialize_model_for_process(self.model)
            serializer = str(model_payload[0])
            chunks = [indexed[i:i+chunk_size] for i in range(0, len(indexed), chunk_size)]
            collected = []
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
                initializer=_process_initializer,
                initargs=(model_payload, self._process_config()),
            ) as ex:
                payloads = [
                    (chunk, bool(use_cache), bool(allow_restriction), on_error)
                    for chunk in chunks
                ]
                for part in ex.map(_process_certify_chunk, payloads):
                    collected.extend(part)
            collected.sort(key=lambda item: item[0])
            results = [r for _, r in collected]
        info = ManyBoxExecutionInfo(
            n_boxes=len(normalized),
            requested_workers=requested_workers,
            resolved_workers=int(workers),
            requested_parallel=requested_parallel,
            resolved_parallel=str(parallel),
            chunk_size=int(chunk_size),
            elapsed_seconds=float(time.perf_counter() - t0),
            model_serializer=serializer,
            machine_rigorous=bool(all(
                getattr((r.result if isinstance(r, ManyBoxCertificationReport) else r),
                        "machine_rigorous", False)
                for r in results if not isinstance(r, ManyBoxCertificationReport) or r.ok
            )),
            scheduling_note=scheduling_note,
            n_succeeded=sum(
                1 for r in results
                if not isinstance(r, ManyBoxCertificationReport) or r.ok
            ),
            n_failed=sum(
                1 for r in results
                if isinstance(r, ManyBoxCertificationReport) and not r.ok
            ),
        )
        with self._lock:
            self._last_manybox_execution = info
        return results

    def certify_many_bounds(self, boxes, **kwargs):
        on_error = _validate_on_error(kwargs.get("on_error", "raise"))
        results = self.certify_many(boxes, **kwargs)
        return _stack_many_bounds(results, on_error)

    def enclose_many(self, boxes, *, workers="auto", parallel="auto", return_info=False, **kwargs):
        results = self.certify_many(boxes, workers=workers, parallel=parallel, **kwargs)
        return (results, self.last_manybox_execution) if return_info else results

    def enclose_many_bounds(self, boxes, *, workers="auto", parallel="auto", return_info=False, **kwargs):
        packed = self.certify_many_bounds(boxes, workers=workers, parallel=parallel, **kwargs)
        if return_info:
            return (*packed, self.last_manybox_execution)
        return packed

    def manybox_executor(self, *, workers="auto", parallel="auto", chunk_size=1,
                         use_cache=True, allow_restriction=True, on_error="raise"):
        """Return a persistent executor for successive box batches."""
        return PreparedManyBoxExecutor(
            self, workers=workers, parallel=parallel, chunk_size=chunk_size,
            use_cache=use_cache, allow_restriction=allow_restriction,
            on_error=on_error,
        )

    def iter_certify_many(self, boxes: Iterable, **kwargs) -> Iterator:
        on_error = _validate_on_error(kwargs.pop("on_error", "raise"))
        use_cache = kwargs.get("use_cache", True)
        allow_restriction = kwargs.get("allow_restriction", True)
        for index, raw in enumerate(boxes):
            box = _normalize_one_box(raw, self.baseline._dimension)
            yield _certify_one_with_policy(
                self, index, box, use_cache=use_cache,
                allow_restriction=allow_restriction, on_error=on_error,
            )

    def iter_bounds_many(self, boxes: Iterable, **kwargs) -> Iterator:
        on_error = _validate_on_error(kwargs.get("on_error", "raise"))
        for item in self.iter_certify_many(boxes, **kwargs):
            if on_error == "raise":
                yield np.asarray(item.lower, dtype=float).copy(), np.asarray(item.upper, dtype=float).copy()
            elif item.ok:
                yield ManyBoxBoundsItemReport(
                    index=item.index, ok=True,
                    lower=np.asarray(item.result.lower, dtype=float).copy(),
                    upper=np.asarray(item.result.upper, dtype=float).copy(),
                )
            else:
                yield ManyBoxBoundsItemReport(index=item.index, ok=False, failure=item.failure)

    def propose_centers_many(self, proposal_model, boxes, *, accelerator="auto", dtype=None):
        from .accelerators import proposal_box_centers
        normalized = _normalize_boxes(boxes, self.baseline._dimension)
        endpoint_boxes = [[(I.lo, I.hi) for I in b] for b in normalized]
        return proposal_box_centers(
            proposal_model, endpoint_boxes, device=accelerator, dtype=dtype
        )




def prepare(model, **kwargs) -> PreparedAutoIA | PreparedConfiguredAutoIA:
    """Prepare an engineering model once for repeated certified queries.

    ``prepare(model)`` is the novice default and creates a lazy automatic
    configuration session with polynomial/order-4 baseline.  Explicit
    ``order=`` or ``representation=`` requests a fixed expert configuration
    unless an explicit ``configuration_policy`` says otherwise.
    """
    order_explicit = "order" in kwargs
    representation_explicit = "representation" in kwargs
    policy = kwargs.pop("configuration_policy", None)
    if policy is None:
        if order_explicit or representation_explicit:
            return PreparedAutoIA(model, **kwargs)
        policy = ConfigurationPolicy(mode="auto")
    if getattr(policy, "mode", None) == "fixed":
        return PreparedAutoIA(model, **kwargs)
    return PreparedConfiguredAutoIA(model, configuration_policy=policy, **kwargs)


__all__ = [
    "PreparedAutoIA", "PreparedConfiguredAutoIA", "PreparedManyBoxExecutor",
    "PreparedSessionStats", "PreparedConfiguredSessionStats", "ManyBoxExecutionInfo",
    "ManyBoxFailure", "ManyBoxCertificationReport", "ManyBoxBoundsItemReport",
    "CertificateCache", "prepare",
]


def enclose_many(model, boxes, *, workers="auto", parallel="auto", return_info=False,
                 on_error="raise", **prepare_options):
    """Prepare ``model`` once and rigorously enclose many independent boxes.

    This is execution infrastructure, not a global optimizer: callers decide
    which boxes to evaluate.  With ``return_info=True`` the function returns
    ``(results, ManyBoxExecutionInfo)`` so applications can record the resolved
    CPU scheduling strategy without affecting proof semantics.
    """
    session = prepare(model, **prepare_options)
    return session.enclose_many(
        boxes, workers=workers, parallel=parallel, return_info=return_info,
        on_error=on_error,
    )


def enclose_many_bounds(model, boxes, *, workers="auto", parallel="auto", return_info=False,
                        on_error="raise", **prepare_options):
    session = prepare(model, **prepare_options)
    return session.enclose_many_bounds(
        boxes, workers=workers, parallel=parallel, return_info=return_info,
        on_error=on_error,
    )
