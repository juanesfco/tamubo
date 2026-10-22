"""Validated semantic primitive for parameter-dependent linear systems.

The established affine path was extracted from the Version-0.11 reference core
during the Step-1 architecture freeze and remains bit-for-bit preserved.  Step 5
adds a separately audited higher-order polynomial solve path with rigorous
predictor/preconditioner portfolios.
"""

from __future__ import annotations

import numpy as np

from ..trusted import (
    Interval, point_interval, abs_upper_interval,
    rigorous_matvec_point, rigorous_matvec_point_interval,
    rigorous_matmul_point, rigorous_matmul_point_interval, interval_matvec,
    nonnegative_matvec_upper, nonnegative_matmul_upper,
    row_sum_norm_upper, matrix_product_abs_upper,
    identity_minus_product_abs_upper, add_nonnegative_matrices_upper,
    scale_nonnegative_matrix_upper,
)
from ..polynomial import (
    zero_alpha, basis_alpha, total_degree, multi_indices_exact_degree,
    monomial_range_symmetric, range_polynomial,
    range_point_polynomial_family,
    pack_interval_polynomial_family, range_packed_interval_termwise,
    interval_objects_from_endpoints, range_interval_polynomial_family,
)
from ..enclosures import (
    ParameterizedEnclosure, IntervalEnclosure, ValidatedValue,
    SharedGeneratorPolynomialEnclosure, extract_generator_polynomial_array,
    contains_pe, extract_affine_array, extract_polynomial_array, contains_validated,
    unwrap_for_polynomial as _unwrap_for_polynomial,
    unwrap_for_generator as _unwrap_for_generator,
    unwrap_for_interval as _unwrap_for_interval,
    all_active_representations as _all_active_representations,
    safe_interval_representative as _safe_interval_representative,
)

# =============================================================================
# Semantic np.linalg.solve primitive
# =============================================================================



def alpha_add(a, b):
    return tuple(int(x) + int(y) for x, y in zip(a, b))


def alpha_sub(a, b):
    return tuple(int(x) - int(y) for x, y in zip(a, b))


def alpha_leq(a, b):
    """Return True when multi-index ``a`` is componentwise <= ``b``."""
    return all(int(x) <= int(y) for x, y in zip(a, b))


def build_solution_polynomials_general(A_coeff, b_coeff, m, order, original_solve=np.linalg.solve, reference_matrix=None):
    """Proposal coefficients for a retained polynomial linear system.

    Step 9 batches all coefficient right-hand sides of the same total degree
    into one ``solve(Aref, RHS)`` call.  A nonzero matrix coefficient always
    lowers total degree, so coefficient equations within one degree are mutually
    independent once all lower degrees are available.  The proposal remains
    nontrusted; the later residual certificate validates the actually stored
    binary64 coefficients.
    """
    z = zero_alpha(m)
    if z not in A_coeff:
        raise ValueError("Polynomial solve requires a constant matrix coefficient A_0.")
    A0 = np.asarray(A_coeff[z], dtype=float)
    if A0.ndim != 2 or A0.shape[0] != A0.shape[1]:
        raise ValueError("A must be square.")
    Aref = A0 if reference_matrix is None else np.asarray(reference_matrix, dtype=float)
    if Aref.shape != A0.shape:
        raise ValueError("reference_matrix must have the same shape as A_0.")
    n = A0.shape[0]

    nonzero_A = [
        (beta, np.asarray(Abeta, dtype=float))
        for beta, Abeta in A_coeff.items()
        if beta != z and total_degree(beta) <= int(order)
    ]

    coeff = {}
    for degree in range(0, int(order) + 1):
        alphas = list(multi_indices_exact_degree(m, degree))
        rhs_columns = []
        for alpha in alphas:
            rhs = np.zeros(n, dtype=float)
            if alpha in b_coeff:
                rhs = np.asarray(b_coeff[alpha], dtype=float).copy()
            for beta, Abeta in nonzero_A:
                if alpha_leq(beta, alpha):
                    gamma = alpha_sub(alpha, beta)
                    rhs -= Abeta @ coeff[gamma]
            rhs_columns.append(rhs)

        try:
            if len(alphas) == 1:
                solved = np.asarray(original_solve(Aref, rhs_columns[0]), dtype=float).reshape(n, 1)
            else:
                RHS = np.column_stack(rhs_columns)
                solved = np.asarray(original_solve(Aref, RHS), dtype=float)
                if solved.shape != RHS.shape:
                    raise ValueError("Batched proposal solve returned an unexpected shape.")
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            raise

        for col, alpha in enumerate(alphas):
            coeff[alpha] = np.asarray(solved[:, col], dtype=float).copy()

    for alpha, vec in coeff.items():
        if np.any(~np.isfinite(vec)):
            raise FloatingPointError(
                f"Nonfinite predictor coefficient at multi-index {alpha}."
            )
    return coeff


def residual_coefficient_intervals_general(A_coeff, b_coeff, X, m, backend):
    """Rigorous interval coefficients of ``b_hat - A_hat P``.

    ``A_coeff`` and ``b_coeff`` contain stored binary64 point coefficient arrays.
    Exact-real products and sums of those stored values are enclosed by the
    trusted reduction kernel.
    """
    z = zero_alpha(m)
    A0 = np.asarray(A_coeff[z], dtype=float)
    n = A0.shape[0]
    R = {}

    def get(alpha):
        if alpha not in R:
            R[alpha] = interval_vector_zeros(n)
        return R[alpha]

    # + b_hat(q)
    for beta, b_beta in b_coeff.items():
        add_interval_vector_inplace(
            get(beta), point_vector_as_intervals(b_beta), backend, +1
        )

    # - A_hat(q) P(q)
    for beta, A_beta in A_coeff.items():
        A_beta = np.asarray(A_beta, dtype=float)
        for alpha, x_alpha in X.items():
            gamma = alpha_add(beta, alpha)
            Ax = rigorous_matvec_point(A_beta, x_alpha, backend)
            add_interval_vector_inplace(get(gamma), Ax, backend, -1)

    return R



def range_point_polynomial_array(coeff_arrays, rem, h, backend, range_method):
    """Rigorously range every entry of a point-polynomial array plus remainder.

    Step 9d lowers the common-support family to numeric buffers for termwise,
    Bernstein, and portfolio ranging.  The portfolio uses packed derivative /
    monotonicity information and the dimension-separable reduced-Bernstein
    transform.  Direct monotonicity retains the historical scalar path for
    bitwise regression compatibility.
    """
    rem = np.asarray(rem, dtype=object)
    out = np.empty(rem.shape, dtype=object)
    meta = np.empty(rem.shape, dtype=object)

    if range_method in {"termwise", "bernstein", "portfolio"}:
        ranges, packed_meta = range_point_polynomial_family(
            coeff_arrays, h, backend, range_method
        )
        if tuple(np.asarray(ranges, dtype=object).shape) != tuple(rem.shape):
            raise ValueError("Packed polynomial family shape does not match remainder shape.")
        for idx in np.ndindex(rem.shape):
            out[idx] = backend.add(ranges[idx], rem[idx])
            meta[idx] = packed_meta[idx]
        return out, meta

    for idx in np.ndindex(rem.shape):
        poly = {
            alpha: float(values[idx])
            for alpha, values in coeff_arrays.items()
            if float(values[idx]) != 0.0
        }
        pr, pmeta = range_polynomial(poly, h, backend, range_method)
        out[idx] = backend.add(pr, rem[idx])
        meta[idx] = pmeta
    return out, meta


def representative_from_interval_array(interval_array):
    """Finite point proposal chosen componentwise inside a rigorous interval array."""
    interval_array = np.asarray(interval_array, dtype=object)
    out = np.empty(interval_array.shape, dtype=float)
    for idx in np.ndindex(interval_array.shape):
        out[idx] = _safe_interval_representative(interval_array[idx])
    return out


def polynomial_preconditioner_portfolio(A_coeff, A_rem, h, backend, range_method,
                                        original_solve):
    """Generate generic point preconditioners for a polynomial matrix family.

    Candidate 1 is the retained constant matrix ``A_0``.  Candidate 2 is the
    componentwise representative of a rigorous interval hull of the full matrix
    family.  Candidate generation is nontrusted; each candidate is accepted or
    rejected solely through the later rigorous defect bound.
    """
    z = zero_alpha(len(h))
    A0 = np.asarray(A_coeff[z], dtype=float)
    candidates = []

    def add_candidate(name, Aref):
        try:
            C = original_solve(Aref, np.eye(Aref.shape[0]))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return
        if np.any(~np.isfinite(C)):
            return
        candidates.append((name, np.asarray(Aref, dtype=float), C))

    add_candidate("constant-coefficient", A0)

    A_hull, A_hull_meta = range_point_polynomial_array(
        A_coeff, A_rem, h, backend, range_method
    )
    A_mid = representative_from_interval_array(A_hull)
    if not np.array_equal(A_mid, A0):
        add_candidate("family-hull-midpoint", A_mid)

    return candidates, A_hull, A_hull_meta, A_mid


def preconditioned_defect_bounds_general(C, A_coeff, A_rem, h, backend, range_method):
    """Rigorous portfolio bounds for ``|I - C A(q)|``.

    Returns ``B`` satisfying ``|I-C A(q)| <= B`` componentwise for the full
    polynomial-plus-remainder matrix family.  Two independent polynomial bounds
    are formed:

    1. an absolute-value-first magnitude sum over monomials;
    2. direct interval-coefficient polynomial ranging of each signed entry of
       ``I - C A_hat(q)``.

    Their componentwise minimum is sound because both bound the same quantity.
    The matrix remainder contribution ``|C| E_A`` is then added rigorously.
    """
    z = zero_alpha(len(h))
    A0 = np.asarray(A_coeff[z], dtype=float)
    n = A0.shape[0]

    # Legacy-style magnitude bound generalized to arbitrary monomials.
    B_legacy_poly = identity_minus_product_abs_upper(C, A0, backend)
    for beta, A_beta in A_coeff.items():
        if beta == z:
            continue
        CA_abs = matrix_product_abs_upper(C, A_beta, backend)
        H = abs_upper_interval(monomial_range_symmetric(beta, h, backend))
        term = scale_nonnegative_matrix_upper(H, CA_abs, backend)
        B_legacy_poly = add_nonnegative_matrices_upper(
            B_legacy_poly, term, backend
        )

    # Direct signed interval-coefficient polynomial for D_hat = I - C A_hat.
    D_coeff = {}
    for beta, A_beta in A_coeff.items():
        CA = rigorous_matmul_point(C, A_beta, backend)
        DI = np.empty(CA.shape, dtype=object)
        for i in range(CA.shape[0]):
            for j in range(CA.shape[1]):
                if beta == z:
                    delta = 1.0 if i == j else 0.0
                    DI[i, j] = backend.sub(point_interval(delta), CA[i, j])
                else:
                    DI[i, j] = backend.sub(point_interval(0.0), CA[i, j])
        D_coeff[beta] = DI

    # Step 9d: lower the entire signed matrix-polynomial family once and
    # batch the proof-critical interval-coefficient reductions across all n^2
    # entries.  The scalar ``range_interval_coefficient_poly`` implementation
    # remains unchanged below as an independent reference oracle.
    D_ranges, D_range_meta = range_interval_polynomial_family(
        D_coeff, h, backend, range_method
    )
    B_direct_poly = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            B_direct_poly[i, j] = abs_upper_interval(D_ranges[i, j])

    B_poly = np.minimum(B_legacy_poly, B_direct_poly)

    E_A = interval_remainder_abs_array(A_rem)
    Cabs = np.abs(C)
    CE = nonnegative_matmul_upper(Cabs, E_A, backend)
    B = add_nonnegative_matrices_upper(B_poly, CE, backend)
    return B, {
        "B_legacy_polynomial": B_legacy_poly,
        "B_direct_polynomial": B_direct_poly,
        "B_polynomial_portfolio": B_poly,
        "D_polynomial_ranges": D_ranges,
        "D_polynomial_range_meta": D_range_meta,
        "CE_remainder": CE,
    }


def build_solution_polynomials(A0, A_i, b0, b_i, order):
    """
    Proposal layer only: vector point coefficients alpha -> x_alpha.

    These solves need not be verified.  The trusted layer certifies the residual of
    the actual stored coefficients returned here.
    """
    m = len(A_i)
    coeff = {}
    z = zero_alpha(m)
    coeff[z] = np.linalg.solve(A0, b0)

    if order >= 1:
        for i in range(m):
            alpha = basis_alpha(m, i)
            coeff[alpha] = np.linalg.solve(A0, b_i[i] - A_i[i] @ coeff[z])

    for degree in range(2, order + 1):
        for alpha in multi_indices_exact_degree(m, degree):
            rhs = np.zeros_like(b0)
            for i in range(m):
                if alpha[i] > 0:
                    beta = list(alpha)
                    beta[i] -= 1
                    beta = tuple(beta)
                    rhs -= A_i[i] @ coeff[beta]
            coeff[alpha] = np.linalg.solve(A0, rhs)

    for alpha, vec in coeff.items():
        if np.any(~np.isfinite(vec)):
            raise FloatingPointError(
                f"Nonfinite predictor coefficient at multi-index {alpha}."
            )
    return coeff


def interval_vector_zeros(n):
    out = np.empty(n, dtype=object)
    for i in range(n):
        out[i] = Interval(0.0, 0.0)
    return out


def add_interval_vector_inplace(target, source, backend, sign=+1):
    for j in range(len(target)):
        if sign == +1:
            target[j] = backend.add(target[j], source[j])
        elif sign == -1:
            target[j] = backend.sub(target[j], source[j])
        else:
            raise ValueError("sign must be +1 or -1")


def point_vector_as_intervals(v):
    v = np.asarray(v, dtype=float)
    out = np.empty(len(v), dtype=object)
    for j, value in enumerate(v):
        out[j] = point_interval(value)
    return out


def residual_coefficient_intervals(A0, A_i, b0, b_i, X, backend):
    """
    Rigorous interval coefficients of

        r_hat(q) = b_hat(q) - A_hat(q) P(q)

    where all A,b,P coefficients are the stored point binary64 values.
    """
    m = len(A_i)
    n = len(b0)
    R = {}

    def get(alpha):
        if alpha not in R:
            R[alpha] = interval_vector_zeros(n)
        return R[alpha]

    # b_hat(q)
    add_interval_vector_inplace(
        get(zero_alpha(m)), point_vector_as_intervals(b0), backend, +1
    )
    for i in range(m):
        add_interval_vector_inplace(
            get(basis_alpha(m, i)), point_vector_as_intervals(b_i[i]), backend, +1
        )

    # -A0 P
    for alpha, xalpha in X.items():
        Ax = rigorous_matvec_point(A0, xalpha, backend)
        add_interval_vector_inplace(get(alpha), Ax, backend, -1)

    # -sum_i q_i A_i P
    for alpha, xalpha in X.items():
        for i in range(m):
            gamma = list(alpha)
            gamma[i] += 1
            gamma = tuple(gamma)
            Aix = rigorous_matvec_point(A_i[i], xalpha, backend)
            add_interval_vector_inplace(get(gamma), Aix, backend, -1)

    return R


def abs_bound_interval_vector_poly(R, h, backend):
    """Componentwise g with |r(q)| <= g for an interval-coefficient polynomial."""
    n = len(next(iter(R.values())))
    ranges = [Interval(0.0, 0.0) for _ in range(n)]

    for alpha, vecI in R.items():
        M = monomial_range_symmetric(alpha, h, backend)
        for j in range(n):
            ranges[j] = backend.add(ranges[j], backend.mul(vecI[j], M))

    g = np.empty(n)
    for j in range(n):
        g[j] = abs_upper_interval(ranges[j])
    return g, ranges


def abs_bound_interval_vector_poly_packed(R, h, backend):
    """Batched Step-9d form of :func:`abs_bound_interval_vector_poly`.

    The legacy scalar function above is intentionally retained as the historical
    bitwise regression oracle.  This lowered form is used only by the general
    polynomial semantic-solve path, where arithmetic reordering is permitted as
    long as the exact-real result remains rigorously enclosed.
    """
    packed = pack_interval_polynomial_family(R, len(h))
    lo, hi = range_packed_interval_termwise(packed, h, backend)
    ranges_array = interval_objects_from_endpoints(lo, hi)
    ranges = [ranges_array[j] for j in range(len(ranges_array))]
    g = np.maximum(np.abs(np.asarray(lo, dtype=float)), np.abs(np.asarray(hi, dtype=float)))
    return np.asarray(g, dtype=float), ranges






def representative_in_interval(I):
    """Choose a finite point coefficient guaranteed to lie in interval I."""
    if I.lo == I.hi:
        return float(I.lo)
    # The representative is only a proposal; rigor comes from enclosing I-rep.
    mid = float(0.5 * I.lo + 0.5 * I.hi)
    if not np.isfinite(mid):
        raise FloatingPointError("Nonfinite interval-coefficient representative.")
    return float(min(max(mid, I.lo), I.hi))


def range_interval_coefficient_poly(polyI, h, backend, method="portfolio"):
    """
    Range an interval-coefficient polynomial rigorously.

    For each coefficient interval I_alpha choose a point representative c_alpha in
    I_alpha, range the point polynomial with the full range portfolio, and add a
    rigorous termwise range of the coefficient uncertainty

        sum_alpha (I_alpha - c_alpha) q^alpha.

    We also compute the direct termwise interval-coefficient range and intersect
    the two sound enclosures.  Therefore this routine cannot be worse than its
    direct termwise bound.
    """
    point_poly = {}
    error_range = Interval(0.0, 0.0)
    direct_range = Interval(0.0, 0.0)

    for alpha, Ialpha in polyI.items():
        rep = representative_in_interval(Ialpha)
        if rep != 0.0:
            point_poly[alpha] = rep

        M = monomial_range_symmetric(alpha, h, backend)

        # Direct interval-coefficient termwise range.
        direct_range = backend.add(
            direct_range, backend.mul(Ialpha, M)
        )

        # Coefficient uncertainty around the stored point representative.
        Ealpha = backend.sub(Ialpha, point_interval(rep))
        error_range = backend.add(
            error_range, backend.mul(Ealpha, M)
        )

    point_range, point_meta = range_polynomial(
        point_poly, h, backend, method
    )
    decomposed = backend.add(point_range, error_range)

    lo = max(direct_range.lo, decomposed.lo)
    hi = min(direct_range.hi, decomposed.hi)
    if lo > hi:
        raise RuntimeError(
            "Disjoint sound interval-coefficient polynomial enclosures; "
            "treat as an internal error."
        )
    return Interval(lo, hi), {
        "direct_termwise": direct_range,
        "point_plus_coefficient_error": decomposed,
        "point_range_meta": point_meta,
    }


def left_multiply_interval_vector_polynomial(C, R, backend):
    """Return interval coefficients of C R(q) for interval-vector polynomial R."""
    out = {}
    for alpha, vecI in R.items():
        out[alpha] = rigorous_matvec_point_interval(C, vecI, backend)
    return out


def range_interval_vector_polynomial(R, h, backend, method="portfolio"):
    """Componentwise ranges of an interval-vector polynomial."""
    n = len(next(iter(R.values())))
    ranges = []
    metas = []
    for j in range(n):
        poly_j = {alpha: vecI[j] for alpha, vecI in R.items()}
        rj, meta = range_interval_coefficient_poly(
            poly_j, h, backend, method
        )
        ranges.append(rj)
        metas.append(meta)
    return ranges, metas




def interval_remainder_abs_array(rem):
    rem = np.asarray(rem, dtype=object)
    out = np.empty(rem.shape)
    for idx in np.ndindex(rem.shape):
        out[idx] = abs_upper_interval(rem[idx])
    return out


def predictor_component_polynomials(X, n):
    out = []
    for j in range(n):
        out.append({alpha: float(vec[j]) for alpha, vec in X.items() if vec[j] != 0.0})
    return out


def predictor_abs_bounds(X, h, backend, range_method, n):
    polys = predictor_component_polynomials(X, n)
    s = np.empty(n)
    ranges = []
    metas = []
    for j, poly in enumerate(polys):
        r, meta = range_polynomial(poly, h, backend, range_method)
        ranges.append(r)
        metas.append(meta)
        s[j] = abs_upper_interval(r)
    return s, ranges, metas


def refine_componentwise_correction(c, B, d_scalar, backend, max_iter=32):
    """
    Rigorously sharpen the scalar correction radius into a componentwise vector.

    Start from the already-proved bound

        |e| <= d_scalar * 1.

    If d is any valid componentwise upper bound, then

        |e| <= c + B |e| <= c + B d.

    Therefore an outward-rounded evaluation of c + B d is another valid upper
    bound.  Intersecting (componentwise minimum) with the previous valid bound is
    also sound.  Repeating approaches the fixed-point bound (I-B)^(-1)c without
    requiring a proof-critical linear solve.
    """
    c = np.asarray(c, dtype=float)
    B = np.asarray(B, dtype=float)
    if c.ndim != 1 or B.shape != (len(c), len(c)):
        raise ValueError("Invalid c/B shapes for componentwise correction.")
    if np.any(c < 0.0) or np.any(B < 0.0) or d_scalar < 0.0:
        raise ValueError("Correction refinement requires nonnegative data.")

    d = np.full(len(c), float(d_scalar))
    history = [d.copy()]

    for _ in range(int(max_iter)):
        Bd = nonnegative_matvec_upper(B, d, backend)
        candidate = np.empty_like(d)
        for j in range(len(d)):
            candidate[j] = backend.add(
                point_interval(c[j]), point_interval(Bd[j])
            ).hi

        new_d = np.minimum(d, candidate)
        history.append(new_d.copy())

        # Equality of stored binary64 values means outward iteration has plateaued.
        if np.array_equal(new_d, d):
            d = new_d
            break
        d = new_d

    return d, history


def constant_solution_predictor(Aref, bref, m, original_solve):
    """Proposal containing only a constant solution coefficient."""
    x0 = original_solve(np.asarray(Aref, dtype=float), np.asarray(bref, dtype=float))
    if np.any(~np.isfinite(x0)):
        raise FloatingPointError("Nonfinite constant solution predictor.")
    return {zero_alpha(m): np.asarray(x0, dtype=float)}


def evaluate_general_polynomial_candidate(A_coeff, b_coeff, A_rem, b_rem,
                                          X, C, B, B_meta, prototype):
    """Evaluate one fully rigorous predictor/preconditioner certificate candidate."""
    P = prototype
    n = np.asarray(A_coeff[zero_alpha(P.m)]).shape[0]

    Rhat = residual_coefficient_intervals_general(
        A_coeff, b_coeff, X, P.m, P.backend
    )
    g, residual_component_ranges = abs_bound_interval_vector_poly_packed(
        Rhat, P.h, P.backend
    )

    s, predictor_ranges, predictor_range_meta = predictor_abs_bounds(
        X, P.h, P.backend, P.range_method, n
    )

    E_A = interval_remainder_abs_array(A_rem)
    e_b = interval_remainder_abs_array(b_rem)
    EA_s = nonnegative_matvec_upper(E_A, s, P.backend)
    rmax = np.empty(n)
    for j in range(n):
        temp = P.backend.add(point_interval(g[j]), point_interval(e_b[j])).hi
        rmax[j] = P.backend.add(
            point_interval(temp), point_interval(EA_s[j])
        ).hi

    Cabs = np.abs(C)
    c_legacy = nonnegative_matvec_upper(Cabs, rmax, P.backend)

    CRhat = left_multiply_interval_vector_polynomial(C, Rhat, P.backend)
    CRhat_ranges_array, CRhat_range_meta_array = range_interval_polynomial_family(
        CRhat, P.h, P.backend, P.range_method
    )
    CRhat_ranges = [CRhat_ranges_array[j] for j in range(len(CRhat_ranges_array))]
    CRhat_range_meta = [
        CRhat_range_meta_array[j] for j in range(len(CRhat_range_meta_array))
    ]
    Cb_rem = rigorous_matvec_point_interval(C, b_rem, P.backend)
    CA_rem = rigorous_matmul_point_interval(C, A_rem, P.backend)
    predictor_range_intervals = np.asarray(predictor_ranges, dtype=object)
    CArem_P = interval_matvec(CA_rem, predictor_range_intervals, P.backend)

    forcing_ranges = []
    c_direct = np.empty(n)
    for j in range(n):
        total = P.backend.add(CRhat_ranges[j], Cb_rem[j])
        total = P.backend.sub(total, CArem_P[j])
        forcing_ranges.append(total)
        c_direct[j] = abs_upper_interval(total)
    c = np.minimum(c_legacy, c_direct)

    rho_upper = row_sum_norm_upper(B, P.backend)
    if not rho_upper < 1.0:
        return {
            "success": False,
            "failure": "contraction",
            "rho_upper": rho_upper,
            "B": B,
            "B_meta": B_meta,
            "X": X,
            "C": C,
        }

    eta_upper = float(np.max(c))
    delta_lower = P.backend.sub(
        point_interval(1.0), point_interval(rho_upper)
    ).lo
    if not delta_lower > 0.0:
        return {
            "success": False,
            "failure": "rounded-contraction-margin",
            "rho_upper": rho_upper,
            "B": B,
            "B_meta": B_meta,
            "X": X,
            "C": C,
        }

    d_upper = P.backend.div(
        point_interval(eta_upper), point_interval(delta_lower)
    ).hi
    d_componentwise, correction_history = refine_componentwise_correction(
        c, B, d_upper, P.backend, max_iter=32
    )

    # Rigorous candidate quality metric: range the predictor and add the proven
    # componentwise correction.  We select among sound candidates by certified
    # total output width, never by an unverified residual surrogate.
    candidate_ranges = []
    candidate_widths = np.empty(n)
    for j in range(n):
        corr = Interval(-float(d_componentwise[j]), float(d_componentwise[j]))
        rj = P.backend.add(predictor_ranges[j], corr)
        candidate_ranges.append(rj)
        candidate_widths[j] = P.backend.sub(
            point_interval(rj.hi), point_interval(rj.lo)
        ).hi
    total_width = 0.0
    for w in candidate_widths:
        total_width = P.backend.add(
            point_interval(total_width), point_interval(float(w))
        ).hi

    return {
        "success": True,
        "X": X,
        "C": C,
        "g": g,
        "s": s,
        "E_A": E_A,
        "e_b": e_b,
        "rmax": rmax,
        "c": c,
        "c_legacy": c_legacy,
        "c_direct": c_direct,
        "CRhat_ranges": CRhat_ranges,
        "CRhat_range_meta": CRhat_range_meta,
        "Cb_rem": Cb_rem,
        "CA_rem": CA_rem,
        "CArem_P": CArem_P,
        "forcing_ranges": forcing_ranges,
        "B": B,
        "B_meta": B_meta,
        "rho_upper": rho_upper,
        "eta_upper": eta_upper,
        "delta_lower": delta_lower,
        "d_upper": d_upper,
        "d_componentwise": d_componentwise,
        "correction_history": correction_history,
        "residual_component_ranges": residual_component_ranges,
        "predictor_ranges": predictor_ranges,
        "predictor_range_meta": predictor_range_meta,
        "candidate_output_ranges": candidate_ranges,
        "candidate_widths": candidate_widths,
        "candidate_total_width": total_width,
    }


def _generator_array_hull(rep_array):
    """Rigorously range an array of shared-generator scalar representations."""
    rep_array = np.asarray(rep_array, dtype=object)
    out = np.empty(rep_array.shape, dtype=object)
    meta = np.empty(rep_array.shape, dtype=object)
    for idx in np.ndindex(rep_array.shape):
        out[idx], meta[idx] = rep_array[idx].range()
    return out, meta


def _generator_preconditioner_portfolio(A_rep, A_coeff, prototype, original_solve):
    """Point preconditioner proposals for the full generator matrix family."""
    P = prototype
    z = zero_alpha(P.m)
    A0 = np.asarray(A_coeff[z], dtype=float)
    A_hull, A_hull_meta = _generator_array_hull(A_rep)
    A_mid = representative_from_interval_array(A_hull)
    candidates = []

    def add(name, Aref):
        Aref = np.asarray(Aref, dtype=float)
        try:
            C = original_solve(Aref, np.eye(Aref.shape[0]))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return
        if np.any(~np.isfinite(C)):
            return
        candidates.append((name, Aref, np.asarray(C, dtype=float)))

    add("constant-coefficient", A0)
    if not np.array_equal(A_mid, A0):
        add("generator-family-hull-midpoint", A_mid)
    return candidates, A_hull, A_hull_meta, A_mid


def _generator_sensitivity_proposal(A_generators, b_generators, X, Aref, original_solve):
    """Proposal ``u_s`` for each incoming auxiliary generator ``eps_s``.

    Step 9 batches all generator forcing columns into one matrix-RHS solve so a
    LAPACK implementation can reuse the factorization of ``Aref``.  These are
    still proposal coefficients only; the joint ``(q, epsilon)`` residual is
    certified afterward.
    """
    first_alpha = next(iter(X))
    z = zero_alpha(len(first_alpha))
    x0 = np.asarray(X[z], dtype=float)
    n = len(x0)
    gids = sorted(set(A_generators) | set(b_generators))
    if not gids:
        return {}

    rhs_columns = []
    for gid in gids:
        A_g = np.asarray(A_generators.get(gid, np.zeros((n, n))), dtype=float)
        b_g = np.asarray(b_generators.get(gid, np.zeros(n)), dtype=float)
        rhs_columns.append(b_g - A_g @ x0)

    try:
        if len(gids) == 1:
            solved = np.asarray(original_solve(np.asarray(Aref, dtype=float), rhs_columns[0]), dtype=float).reshape(n, 1)
        else:
            RHS = np.column_stack(rhs_columns)
            solved = np.asarray(original_solve(np.asarray(Aref, dtype=float), RHS), dtype=float)
            if solved.shape != RHS.shape:
                return {}
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return {}

    out = {}
    for col, gid in enumerate(gids):
        u = np.asarray(solved[:, col], dtype=float).copy()
        if np.any(~np.isfinite(u)):
            continue
        out[int(gid)] = u
    return out


def _generator_predictor_vector(X, U, prototype):
    """Build scalar generator representations for a vector solve predictor."""
    P = prototype
    z = zero_alpha(P.m)
    n = len(np.asarray(X[z]))
    out = np.empty(n, dtype=object)
    for j in range(n):
        coeff = {
            alpha: float(vec[j])
            for alpha, vec in X.items()
            if float(vec[j]) != 0.0
        }
        generators = {
            int(gid): float(vec[j])
            for gid, vec in U.items()
            if float(vec[j]) != 0.0
        }
        out[j] = SharedGeneratorPolynomialEnclosure(
            coeff, Interval(0.0, 0.0), P.h, P.order, P.backend, P.range_method,
            generators=generators, lift_remainder=False,
        )
    return out


def _generator_linear_combination(weights, values, prototype):
    """Rigorous point-weighted linear combination preserving generator ids."""
    total = prototype.constant_like(0.0)
    for weight, value in zip(np.asarray(weights, dtype=float), values):
        if float(weight) != 0.0:
            total = total + value * float(weight)
    return total


def _generator_residual_vector(A_rep, b_rep, x_pred, prototype):
    A_rep = np.asarray(A_rep, dtype=object)
    b_rep = np.asarray(b_rep, dtype=object)
    n = len(b_rep)
    out = np.empty(n, dtype=object)
    for i in range(n):
        ri = b_rep[i].copy()
        for k in range(n):
            ri = ri - A_rep[i, k] * x_pred[k]
        out[i] = ri
    return out


def _generator_defect_bounds(C, A_rep, A_hull, prototype):
    """Rigorously bound ``|I-C A(q,eps)|`` while retaining generator provenance."""
    P = prototype
    A_rep = np.asarray(A_rep, dtype=object)
    n = A_rep.shape[0]
    B_direct = np.empty((n, n), dtype=float)
    D_ranges = np.empty((n, n), dtype=object)
    D_meta = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            d = P.constant_like(1.0 if i == j else 0.0)
            for k in range(n):
                if float(C[i, k]) != 0.0:
                    d = d - A_rep[k, j] * float(C[i, k])
            r, meta = d.range()
            D_ranges[i, j] = r
            D_meta[i, j] = meta
            B_direct[i, j] = abs_upper_interval(r)

    CA_hull = rigorous_matmul_point_interval(C, A_hull, P.backend)
    B_hull = np.empty((n, n), dtype=float)
    D_hull = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            delta = point_interval(1.0 if i == j else 0.0)
            dij = P.backend.sub(delta, CA_hull[i, j])
            D_hull[i, j] = dij
            B_hull[i, j] = abs_upper_interval(dij)

    B = np.minimum(B_direct, B_hull)
    return B, {
        "B_direct_generator": B_direct,
        "B_interval_hull": B_hull,
        "B_generator_portfolio": B,
        "D_generator_ranges": D_ranges,
        "D_generator_range_meta": D_meta,
        "D_interval_hull": D_hull,
    }


def _evaluate_generator_solve_candidate(A_rep, b_rep, X, U, C, A_hull, prototype):
    """Rigorous certificate for one polynomial/generator predictor candidate."""
    P = prototype
    x_pred = _generator_predictor_vector(X, U, P)
    residual = _generator_residual_vector(A_rep, b_rep, x_pred, P)
    n = len(residual)

    residual_ranges = np.empty(n, dtype=object)
    residual_meta = np.empty(n, dtype=object)
    residual_abs = np.empty(n, dtype=float)
    for i in range(n):
        residual_ranges[i], residual_meta[i] = residual[i].range()
        residual_abs[i] = abs_upper_interval(residual_ranges[i])

    c_legacy = nonnegative_matvec_upper(np.abs(C), residual_abs, P.backend)
    Cr = np.empty(n, dtype=object)
    Cr_ranges = np.empty(n, dtype=object)
    Cr_meta = np.empty(n, dtype=object)
    c_direct = np.empty(n, dtype=float)
    for i in range(n):
        Cr[i] = _generator_linear_combination(C[i, :], residual, P)
        Cr_ranges[i], Cr_meta[i] = Cr[i].range()
        c_direct[i] = abs_upper_interval(Cr_ranges[i])
    c = np.minimum(c_legacy, c_direct)

    B, B_meta = _generator_defect_bounds(C, A_rep, A_hull, P)
    rho_upper = row_sum_norm_upper(B, P.backend)
    if not rho_upper < 1.0:
        return {
            "success": False, "failure": "contraction",
            "rho_upper": rho_upper, "B": B, "B_meta": B_meta,
            "X": X, "U": U, "C": C,
        }

    eta_upper = float(np.max(c))
    delta_lower = P.backend.sub(
        point_interval(1.0), point_interval(rho_upper)
    ).lo
    if not delta_lower > 0.0:
        return {
            "success": False, "failure": "rounded-contraction-margin",
            "rho_upper": rho_upper, "B": B, "B_meta": B_meta,
            "X": X, "U": U, "C": C,
        }

    d_upper = P.backend.div(
        point_interval(eta_upper), point_interval(delta_lower)
    ).hi
    d_componentwise, correction_history = refine_componentwise_correction(
        c, B, d_upper, P.backend, max_iter=32
    )

    predictor_ranges = []
    predictor_meta = []
    candidate_ranges = []
    candidate_widths = np.empty(n, dtype=float)
    for j in range(n):
        pr, pm = x_pred[j].range()
        predictor_ranges.append(pr)
        predictor_meta.append(pm)
        corr = Interval(-float(d_componentwise[j]), float(d_componentwise[j]))
        rr = P.backend.add(pr, corr)
        candidate_ranges.append(rr)
        candidate_widths[j] = P.backend.sub(
            point_interval(rr.hi), point_interval(rr.lo)
        ).hi
    total_width = 0.0
    for width in candidate_widths:
        total_width = P.backend.add(
            point_interval(total_width), point_interval(float(width))
        ).hi

    preserved = tuple(sorted(
        gid for gid, vec in U.items()
        if np.any(np.asarray(vec, dtype=float) != 0.0)
    ))
    return {
        "success": True,
        "X": X, "U": U, "C": C,
        "x_predictor": x_pred,
        "residual_generator_values": residual,
        "residual_ranges": residual_ranges,
        "residual_range_meta": residual_meta,
        "c_legacy": c_legacy,
        "Cr_generator_values": Cr,
        "Cr_ranges": Cr_ranges,
        "Cr_range_meta": Cr_meta,
        "c_direct": c_direct,
        "c": c,
        "B": B, "B_meta": B_meta,
        "rho_upper": rho_upper,
        "eta_upper": eta_upper,
        "delta_lower": delta_lower,
        "d_upper": d_upper,
        "d_componentwise": d_componentwise,
        "correction_history": correction_history,
        "predictor_ranges": predictor_ranges,
        "predictor_range_meta": predictor_meta,
        "candidate_output_ranges": candidate_ranges,
        "candidate_widths": candidate_widths,
        "candidate_total_width": total_width,
        "preserved_input_generator_ids": preserved,
    }


def generator_semantic_solve(A, b, prototype, original_solve):
    """Generator-aware validated solve for ``A(q,eps) x = b(q,eps)``.

    The predictor ``P(q) + sum_s u_s eps_s`` reuses incoming generator ids.
    Its actual stored residual is certified over the joint physical/generator
    domain.  The remaining componentwise correction is added using independent
    fresh generators because the fixed-point theorem proves only a box for that
    residual error.
    """
    P = prototype
    A_rep = _unwrap_for_generator(A, P)
    b_rep = _unwrap_for_generator(b, P)
    A_coeff, A_generators, A_rem = extract_generator_polynomial_array(A_rep, P, "A")
    b_coeff, b_generators, b_rem = extract_generator_polynomial_array(b_rep, P, "b")
    z = zero_alpha(P.m)
    A0 = np.asarray(A_coeff[z], dtype=float)
    b0 = np.asarray(b_coeff[z], dtype=float)
    if A0.ndim != 2 or A0.shape[0] != A0.shape[1]:
        raise ValueError("A must be square.")
    if b0.ndim != 1 or A0.shape[0] != len(b0):
        raise ValueError("A and b dimensions do not match.")
    n = len(b0)

    preconditioners, A_hull, A_hull_meta, A_mid = _generator_preconditioner_portfolio(
        A_rep, A_coeff, P, original_solve
    )
    if not preconditioners:
        raise RuntimeError(
            "Generator-aware semantic solve could not construct a finite point preconditioner."
        )
    b_hull, b_hull_meta = _generator_array_hull(b_rep)
    b_mid = representative_from_interval_array(b_hull)

    predictor_candidates = []
    try:
        X = build_solution_polynomials_general(
            A_coeff, b_coeff, P.m, P.order, original_solve=original_solve
        )
        predictor_candidates.append(("formal-A0-recurrence", A0, X))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        pass
    try:
        X = build_solution_polynomials_general(
            A_coeff, b_coeff, P.m, P.order, original_solve=original_solve,
            reference_matrix=A_mid,
        )
        predictor_candidates.append(("generator-family-midpoint-recurrence", A_mid, X))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        pass
    try:
        X = constant_solution_predictor(A_mid, b_mid, P.m, original_solve)
        predictor_candidates.append(("generator-family-midpoint-constant", A_mid, X))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        pass
    if not predictor_candidates:
        raise RuntimeError("Generator-aware semantic solve could not construct a finite predictor.")

    input_gids = tuple(sorted(set(A_generators) | set(b_generators)))
    evaluated = []
    for pred_name, pred_Aref, X in predictor_candidates:
        U_linear = _generator_sensitivity_proposal(
            A_generators, b_generators, X, pred_Aref, original_solve
        )
        generator_predictors = [("first-order-shared-generator", U_linear)]
        if input_gids:
            generator_predictors.append(("zero-generator", {}))
        for gen_name, U in generator_predictors:
            for prec_name, Aref, C in preconditioners:
                cand = _evaluate_generator_solve_candidate(
                    A_rep, b_rep, X, U, C, A_hull, P
                )
                cand["predictor_name"] = pred_name
                cand["generator_predictor_name"] = gen_name
                cand["preconditioner_name"] = prec_name
                cand["preconditioner_reference_matrix"] = Aref
                evaluated.append(cand)

    successful = [c for c in evaluated if c.get("success")]
    if not successful:
        rho_best = min(float(c.get("rho_upper", np.inf)) for c in evaluated)
        raise RuntimeError(
            "Generator-aware semantic solve contraction failed for every candidate; "
            f"best rigorous rho_upper={rho_best:.17g}."
        )

    chosen = min(
        successful,
        key=lambda c: (
            float(np.sum(c["d_componentwise"])),
            float(np.max(c["d_componentwise"])),
            -len(c.get("preserved_input_generator_ids", ())),
            float(c["candidate_total_width"]),
            float(c["rho_upper"]),
        ),
    )

    X = chosen["X"]
    U = chosen["U"]
    d = chosen["d_componentwise"]
    out = np.empty(n, dtype=object)
    for j in range(n):
        coeff = {
            alpha: float(vec[j]) for alpha, vec in X.items()
            if float(vec[j]) != 0.0
        }
        generators = {
            int(gid): float(vec[j]) for gid, vec in U.items()
            if float(vec[j]) != 0.0
        }
        out[j] = SharedGeneratorPolynomialEnclosure(
            coeff, Interval(-float(d[j]), float(d[j])),
            P.h, P.order, P.backend, P.range_method, generators=generators,
        )

    correction_ids = []
    input_gid_set = set(input_gids)
    for value in out:
        correction_ids.extend(g for g in value.generator_ids if g not in input_gid_set)

    cert = dict(chosen)
    cert.update({
        "representation": "generator-polynomial",
        "solve_mode": "generator-aware",
        "generator_aware": True,
        "input_generator_ids": input_gids,
        "A_generator_coefficients": A_generators,
        "b_generator_coefficients": b_generators,
        "A_generator_remainders": A_rem,
        "b_generator_remainders": b_rem,
        "A_family_hull": A_hull,
        "A_family_hull_meta": A_hull_meta,
        "A_family_midpoint": A_mid,
        "b_family_hull": b_hull,
        "b_family_hull_meta": b_hull_meta,
        "b_family_midpoint": b_mid,
        "portfolio_candidates": evaluated,
        "selected_predictor": chosen["predictor_name"],
        "selected_generator_predictor": chosen["generator_predictor_name"],
        "selected_preconditioner": chosen["preconditioner_name"],
        "generator_predictor_coefficients": U,
        "preserved_input_generator_ids": chosen.get("preserved_input_generator_ids", ()),
        "independent_correction_generator_ids": tuple(sorted(set(correction_ids))),
        "generatorized_output_correction": True,
    })
    return out, cert


def polynomial_semantic_solve_general(A, b, prototype, original_solve):
    """Rigorous semantic solve for higher-order polynomial retained inputs.

    This path accepts arbitrary retained polynomial dependence in every entry of
    ``A`` and ``b`` (up to the representation's current truncation order), plus
    the existing rigorous interval remainders.

    Proposal portfolio
    ------------------
    * formal coefficient recurrence using ``A_0`` when available;
    * the same recurrence using the rigorous family-hull midpoint as a proposal
      reference matrix;
    * a constant predictor from that midpoint family representative.

    Preconditioner portfolio
    ------------------------
    * inverse of ``A_0`` when finite;
    * inverse of the rigorous matrix-family hull midpoint when finite.

    Every predictor/preconditioner pair is independently certified.  Among the
    successful pairs AutoIA returns the one with the smallest rigorous total
    output enclosure width.  Thus the portfolio changes sharpness/robustness but
    not the mathematical inclusion guarantee.
    """
    P = prototype
    A_coeff, A_rem = extract_polynomial_array(A, P, "A")
    b_coeff, b_rem = extract_polynomial_array(b, P, "b")
    z = zero_alpha(P.m)
    A0 = np.asarray(A_coeff[z], dtype=float)
    b0 = np.asarray(b_coeff[z], dtype=float)

    if A0.ndim != 2 or A0.shape[0] != A0.shape[1]:
        raise ValueError("A must be square.")
    if b0.ndim != 1 or A0.shape[0] != len(b0):
        raise ValueError("A and b dimensions do not match.")
    n = len(b0)

    preconditioners, A_hull, A_hull_meta, A_mid = polynomial_preconditioner_portfolio(
        A_coeff, A_rem, P.h, P.backend, P.range_method, original_solve
    )
    if not preconditioners:
        raise RuntimeError(
            "Polynomial semantic solve could not construct a finite point "
            "preconditioner from A_0 or the rigorous family-hull midpoint."
        )

    b_hull, b_hull_meta = range_point_polynomial_array(
        b_coeff, b_rem, P.h, P.backend, P.range_method
    )
    b_mid = representative_from_interval_array(b_hull)

    predictor_candidates = []
    try:
        Xformal = build_solution_polynomials_general(
            A_coeff, b_coeff, P.m, P.order, original_solve=original_solve
        )
        predictor_candidates.append(("formal-A0-recurrence", Xformal))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        pass

    # Midpoint-reference recurrence is a proposal only, so it remains sound even
    # though it does not satisfy the formal A_0 coefficient equations exactly.
    try:
        Xmid = build_solution_polynomials_general(
            A_coeff, b_coeff, P.m, P.order,
            original_solve=original_solve, reference_matrix=A_mid
        )
        predictor_candidates.append(("family-midpoint-recurrence", Xmid))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        pass

    try:
        Xconst = constant_solution_predictor(A_mid, b_mid, P.m, original_solve)
        predictor_candidates.append(("family-midpoint-constant", Xconst))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        pass

    if not predictor_candidates:
        raise RuntimeError(
            "Polynomial semantic solve could not construct any finite predictor proposal."
        )

    evaluated = []
    for pred_name, X in predictor_candidates:
        for prec_name, Aref, C in preconditioners:
            B, Bmeta = preconditioned_defect_bounds_general(
                C, A_coeff, A_rem, P.h, P.backend, P.range_method
            )
            Bmeta = dict(Bmeta)
            Bmeta["preconditioner_name"] = prec_name
            cand = evaluate_general_polynomial_candidate(
                A_coeff, b_coeff, A_rem, b_rem, X, C, B, Bmeta, P
            )
            cand["predictor_name"] = pred_name
            cand["preconditioner_name"] = prec_name
            cand["preconditioner_reference_matrix"] = Aref
            evaluated.append(cand)

    successful = [c for c in evaluated if c.get("success")]
    if not successful:
        rho_best = min(float(c.get("rho_upper", np.inf)) for c in evaluated)
        raise RuntimeError(
            "Polynomial semantic solve contraction failed for every generic "
            f"predictor/preconditioner candidate; best rigorous rho_upper={rho_best:.17g}. "
            "Subdivision or a stronger future certifier is required."
        )

    # Preserve useful dependence for downstream code by minimizing the proven
    # unstructured correction first.  A narrow immediate hull obtained by a
    # nearly constant predictor can otherwise discard most of the parameter
    # structure.  Certified output width and contraction are secondary
    # tie-breakers among candidates with similarly small correction remainders.
    chosen = min(
        successful,
        key=lambda c: (
            float(np.sum(c["d_componentwise"])),
            float(np.max(c["d_componentwise"])),
            float(c["candidate_total_width"]),
            float(c["rho_upper"]),
        ),
    )

    X = chosen["X"]
    d_componentwise = chosen["d_componentwise"]
    out = np.empty(n, dtype=object)
    for j in range(n):
        scalar_coeff = {
            alpha: float(vec[j])
            for alpha, vec in X.items()
            if vec[j] != 0.0
        }
        out[j] = ParameterizedEnclosure(
            scalar_coeff,
            Interval(-float(d_componentwise[j]), float(d_componentwise[j])),
            P.h, P.order, P.backend, P.range_method,
        )

    certificate = dict(chosen)
    certificate.update({
        "solve_mode": "polynomial-general",
        "solve_input_polynomial_degree_A": max(total_degree(a) for a in A_coeff),
        "solve_input_polynomial_degree_b": max(total_degree(a) for a in b_coeff),
        "A_polynomial_coefficients": A_coeff,
        "b_polynomial_coefficients": b_coeff,
        "A_family_hull": A_hull,
        "A_family_hull_meta": A_hull_meta,
        "A_family_midpoint": A_mid,
        "b_family_hull": b_hull,
        "b_family_hull_meta": b_hull_meta,
        "b_family_midpoint": b_mid,
        "portfolio_candidates": evaluated,
        "selected_predictor": chosen["predictor_name"],
        "selected_preconditioner": chosen["preconditioner_name"],
    })
    return out, certificate












class SolveContext:
    def __init__(self, prototype):
        self.prototype = prototype
        self.original_solve = np.linalg.solve
        self.certificates = []

    def traced_solve(self, A, b):
        if not (contains_pe(A) or contains_pe(b)):
            return self.original_solve(A, b)

        P = self.prototype

        # Step 5: route higher-order retained polynomial matrices/right-hand
        # sides to the general polynomial certifier.  The affine path below is
        # intentionally unchanged for bit-for-bit no-regression behavior.
        A_poly_coeff, _ = extract_polynomial_array(A, P, "A")
        b_poly_coeff, _ = extract_polynomial_array(b, P, "b")
        if any(
            total_degree(alpha) > 1
            for alpha in set(A_poly_coeff) | set(b_poly_coeff)
        ):
            out, cert = polynomial_semantic_solve_general(
                A, b, P, self.original_solve
            )
            self.certificates.append(cert)
            return out

        A0, A_i, A_rem = extract_affine_array(A, P, "A")
        b0, b_i, b_rem = extract_affine_array(b, P, "b")

        if A0.ndim != 2 or A0.shape[0] != A0.shape[1]:
            raise ValueError("A must be square.")
        if b0.ndim != 1:
            raise ValueError("Version 0.6.1 expects a vector right-hand side.")
        if A0.shape[0] != len(b0):
            raise ValueError("A and b dimensions do not match.")

        n = len(b0)

        # ---------------------------------------------------------------------
        # Proposal layer: unrestricted ordinary floating point.
        # ---------------------------------------------------------------------
        X = build_solution_polynomials(A0, A_i, b0, b_i, P.order)
        C = self.original_solve(A0, np.eye(n))
        if np.any(~np.isfinite(C)):
            raise FloatingPointError("Nonfinite approximate inverse/preconditioner C.")

        # ---------------------------------------------------------------------
        # Trusted layer.
        # ---------------------------------------------------------------------

        # 1) Bound the point-affine residual of the ACTUAL stored predictor.
        Rhat = residual_coefficient_intervals(
            A0, A_i, b0, b_i, X, P.backend
        )
        g, residual_component_ranges = abs_bound_interval_vector_poly(
            Rhat, P.h, P.backend
        )

        # 2) Bound |P(q)| componentwise.
        s, predictor_ranges, predictor_range_meta = predictor_abs_bounds(
            X, P.h, P.backend, P.range_method, n
        )

        # 3) Add affine-input remainder effects:
        #       |r| <= g + e_b + E_A s.
        E_A = interval_remainder_abs_array(A_rem)
        e_b = interval_remainder_abs_array(b_rem)
        EA_s = nonnegative_matvec_upper(E_A, s, P.backend)

        rmax = np.empty(n)
        for j in range(n):
            temp = P.backend.add(point_interval(g[j]), point_interval(e_b[j])).hi
            rmax[j] = P.backend.add(point_interval(temp), point_interval(EA_s[j])).hi

        # 4) Two independent rigorous bounds for c >= |C r|.
        #
        # Legacy v0.6 path:
        #       |C r| <= |C| rmax.
        Cabs = np.abs(C)
        c_legacy = nonnegative_matvec_upper(Cabs, rmax, P.backend)

        # Sharper v0.6.1 path:
        #   Apply C to the interval-coefficient residual polynomial BEFORE
        #   destroying its signed/componentwise structure with absolute values.
        #
        #       C r = C r_hat + C Delta_b - (C Delta_A) P.
        #
        # Each term below is ranged as an interval, then combined before the
        # final absolute value is taken.
        CRhat = left_multiply_interval_vector_polynomial(C, Rhat, P.backend)
        CRhat_ranges, CRhat_range_meta = range_interval_vector_polynomial(
            CRhat, P.h, P.backend, P.range_method
        )

        Cb_rem = rigorous_matvec_point_interval(C, b_rem, P.backend)
        CA_rem = rigorous_matmul_point_interval(C, A_rem, P.backend)
        predictor_range_intervals = np.asarray(predictor_ranges, dtype=object)
        CArem_P = interval_matvec(
            CA_rem, predictor_range_intervals, P.backend
        )

        forcing_ranges = []
        c_direct = np.empty(n)
        for j in range(n):
            total = P.backend.add(CRhat_ranges[j], Cb_rem[j])
            total = P.backend.sub(total, CArem_P[j])
            forcing_ranges.append(total)
            c_direct[j] = abs_upper_interval(total)

        # Both are sound upper bounds on the same quantity.  Their componentwise
        # minimum is therefore also sound and gives a no-regression portfolio.
        c = np.minimum(c_legacy, c_direct)

        # 5) B >= |I - C A(q)|:
        #       |I-CA0| + sum_i h_i |C Ai| + |C| E_A.
        B = identity_minus_product_abs_upper(C, A0, P.backend)

        for i in range(P.m):
            CAi_abs = matrix_product_abs_upper(C, A_i[i], P.backend)
            term = scale_nonnegative_matrix_upper(P.h[i], CAi_abs, P.backend)
            B = add_nonnegative_matrices_upper(B, term, P.backend)

        CE = nonnegative_matmul_upper(Cabs, E_A, P.backend)
        B = add_nonnegative_matrices_upper(B, CE, P.backend)

        # 6) Rigorous infinity-norm upper bound and scalar correction.
        rho_upper = row_sum_norm_upper(B, P.backend)
        if not rho_upper < 1.0:
            raise RuntimeError(
                f"Semantic solve contraction failed: rigorous rho_upper={rho_upper:.17g} >= 1. "
                "A future generic controller should subdivide or switch certifiers."
            )

        eta_upper = float(np.max(c))
        delta_lower = P.backend.sub(
            point_interval(1.0), point_interval(rho_upper)
        ).lo
        if not delta_lower > 0.0:
            raise RuntimeError(
                "Contraction margin lost under outward rounding: "
                f"down(1-rho_upper)={delta_lower:.17g}."
            )

        d_upper = P.backend.div(
            point_interval(eta_upper), point_interval(delta_lower)
        ).hi

        # 7) Sharpen the common scalar radius into a componentwise rigorous
        #    correction vector without a verified linear solve.
        d_componentwise, correction_history = refine_componentwise_correction(
            c, B, d_upper, P.backend, max_iter=32
        )

        # 8) Return common representation with component-specific corrections.
        out = np.empty(n, dtype=object)
        for j in range(n):
            scalar_coeff = {
                alpha: float(vec[j])
                for alpha, vec in X.items()
                if vec[j] != 0.0
            }
            out[j] = ParameterizedEnclosure(
                scalar_coeff,
                Interval(-float(d_componentwise[j]), float(d_componentwise[j])),
                P.h, P.order, P.backend, P.range_method,
            )

        # Lightweight proof diagnostics attached to this context.  They are not
        # required by user model code, but are useful during development/tests.
        certificate = {
            "g": g,
            "s": s,
            "E_A": E_A,
            "e_b": e_b,
            "rmax": rmax,
            "c": c,
            "c_legacy": c_legacy,
            "c_direct": c_direct,
            "CRhat_ranges": CRhat_ranges,
            "CRhat_range_meta": CRhat_range_meta,
            "Cb_rem": Cb_rem,
            "CA_rem": CA_rem,
            "CArem_P": CArem_P,
            "forcing_ranges": forcing_ranges,
            "B": B,
            "rho_upper": rho_upper,
            "eta_upper": eta_upper,
            "delta_lower": delta_lower,
            "d_upper": d_upper,
            "d_componentwise": d_componentwise,
            "correction_history": correction_history,
            "residual_component_ranges": residual_component_ranges,
            "predictor_ranges": predictor_ranges,
            "predictor_range_meta": predictor_range_meta,
        }
        self.certificates.append(certificate)
        return out

    def __enter__(self):
        # dev7: context lifetime collects certificates only.  It no longer mutates
        # process-global NumPy state; user-model dispatch is handled by
        # semantic.dispatch.execute_model_with_dispatch().
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False



# =============================================================================
# Version 0.10 representation-independent semantic solve support
# =============================================================================


def _interval_matrix_midpoint(AI):
    AI = np.asarray(AI, dtype=object)
    A0 = np.empty(AI.shape)
    for idx in np.ndindex(AI.shape):
        A0[idx] = _safe_interval_representative(AI[idx])
    return A0


def _interval_vector_midpoint(bI):
    bI = np.asarray(bI, dtype=object)
    b0 = np.empty(bI.shape)
    for idx in np.ndindex(bI.shape):
        b0[idx] = _safe_interval_representative(bI[idx])
    return b0


def interval_semantic_solve(AI, bI, prototype, original_solve):
    """
    Rigorous semantic linear solve for dependency-blind interval A and b.

    This certifies every matrix/right-hand side in the independently boxed family.
    It therefore may fail even when the original parameterized family is perfectly
    nonsingular; that failure is an informative consequence of representation loss.
    """
    AI = np.asarray(AI, dtype=object)
    bI = np.asarray(bI, dtype=object)
    backend = prototype.backend

    if AI.ndim != 2 or AI.shape[0] != AI.shape[1]:
        raise ValueError("A must be square.")
    if bI.ndim != 1 or AI.shape[0] != len(bI):
        raise ValueError("A and b dimensions do not match.")

    n = len(bI)
    A0 = _interval_matrix_midpoint(AI)
    b0 = _interval_vector_midpoint(bI)

    # Proposal calculations only.
    x0 = original_solve(A0, b0)
    C = original_solve(A0, np.eye(n))
    if np.any(~np.isfinite(x0)) or np.any(~np.isfinite(C)):
        raise FloatingPointError("Nonfinite interval-solve proposal/preconditioner.")

    # Trusted residual r = b - A x0.
    x0I = np.asarray([point_interval(v) for v in x0], dtype=object)
    Ax0 = interval_matvec(AI, x0I, backend)
    rI = np.empty(n, dtype=object)
    for j in range(n):
        rI[j] = backend.sub(bI[j], Ax0[j])

    CrI = rigorous_matvec_point_interval(C, rI, backend)
    c = np.asarray([abs_upper_interval(v) for v in CrI], dtype=float)

    # Trusted D = I - C A and B >= |D|.
    CAI = rigorous_matmul_point_interval(C, AI, backend)
    B = np.empty((n, n))
    DI = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            delta = 1.0 if i == j else 0.0
            DI[i, j] = backend.sub(point_interval(delta), CAI[i, j])
            B[i, j] = abs_upper_interval(DI[i, j])

    rho_upper = row_sum_norm_upper(B, backend)
    if not rho_upper < 1.0:
        raise RuntimeError(
            "Interval semantic solve contraction failed: "
            f"rigorous rho_upper={rho_upper:.17g} >= 1.  "
            "The interval representation has discarded parameter dependence; "
            "switch to the polynomial representation or subdivide."
        )

    eta_upper = float(np.max(c))
    delta_lower = backend.sub(
        point_interval(1.0), point_interval(rho_upper)
    ).lo
    if not delta_lower > 0.0:
        raise RuntimeError("Contraction margin lost under outward rounding.")

    d_scalar = backend.div(
        point_interval(eta_upper), point_interval(delta_lower)
    ).hi
    d_componentwise, history = refine_componentwise_correction(
        c, B, d_scalar, backend, max_iter=32
    )

    out = np.empty(n, dtype=object)
    for j in range(n):
        corr = Interval(-float(d_componentwise[j]), float(d_componentwise[j]))
        xI = backend.add(point_interval(float(x0[j])), corr)
        out[j] = IntervalEnclosure(
            xI, prototype.h, prototype.order,
            backend, prototype.range_method,
        )

    certificate = {
        "representation": "interval",
        "A_interval": AI,
        "b_interval": bI,
        "A0_proposal": A0,
        "b0_proposal": b0,
        "x0_proposal": x0,
        "residual_interval": rI,
        "Cr_interval": CrI,
        "c": c,
        "D_interval": DI,
        "B": B,
        "rho_upper": rho_upper,
        "d_scalar": d_scalar,
        "d_componentwise": d_componentwise,
        "correction_history": history,
    }
    return out, certificate


class GenericSolveContext:
    """Representation-aware semantic dispatcher for np.linalg.solve."""

    def __init__(self, prototype):
        self.prototype = prototype
        self.original_solve = np.linalg.solve
        self.certificates = []

    def traced_solve(self, A, b):
        if not (contains_validated(A) or contains_validated(b)):
            return self.original_solve(A, b)

        # NumPy supports multiple right-hand sides B with shape (n, k).  The
        # linear-system theorem is columnwise, so certify each right-hand side
        # independently while preserving the standard output shape (n, k).
        b_obj = np.asarray(b, dtype=object)
        if b_obj.ndim == 2:
            columns = [self.traced_solve(A, b_obj[:, j]) for j in range(b_obj.shape[1])]
            return np.column_stack(columns)
        if b_obj.ndim != 1:
            raise ValueError(
                "Validated np.linalg.solve currently supports b with shape (n,) or (n,k)."
            )

        P = self.prototype
        kinds = _all_active_representations((A, b))

        # Step 7b: generator-valued inputs are now certified directly over the
        # joint physical/generator domain.  Incoming generator ids are retained
        # in the solution predictor whenever the rigorous residual certificate
        # accepts them.  Only the final componentwise correction receives fresh,
        # independent generator ids because the fixed-point theorem proves a box
        # for that residual error, not a coupled error law.
        if P.representation_name == "generator-polynomial" or "generator-polynomial" in kinds:
            generator_prototype = P.to_generator()
            out_rep, cert = generator_semantic_solve(
                A, b, generator_prototype, self.original_solve
            )
            self.certificates.append(cert)
            return np.asarray([ValidatedValue(v) for v in out_rep], dtype=object)

        # Polynomial-family inputs without active generators retain the mature
        # affine/general polynomial semantic solve paths unchanged.
        if "polynomial" in kinds:
            A_poly = _unwrap_for_polynomial(A, P)
            b_poly = _unwrap_for_polynomial(b, P)
            poly_prototype = P.to_polynomial()

            ctx = SolveContext(poly_prototype)
            ctx.original_solve = self.original_solve
            out_rep = ctx.traced_solve(A_poly, b_poly)
            wrapped = [ValidatedValue(v) for v in out_rep]
            self.certificates.extend(ctx.certificates)
            return np.asarray(wrapped, dtype=object)

        # Otherwise the active uncertain representation is interval-only.
        AI = _unwrap_for_interval(A, P)
        bI = _unwrap_for_interval(b, P)
        interval_prototype = P.to_interval()
        out_rep, cert = interval_semantic_solve(
            AI, bI, interval_prototype, self.original_solve
        )
        self.certificates.append(cert)
        return np.asarray([ValidatedValue(v) for v in out_rep], dtype=object)

    def __enter__(self):
        # dev7: context lifetime collects certificates only.  It no longer mutates
        # process-global NumPy state; user-model dispatch is handled by
        # semantic.dispatch.execute_model_with_dispatch().
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


