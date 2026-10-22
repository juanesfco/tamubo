"""Rigorous recentering of stored point-coefficient polynomials.

This module supports prepared-session certificate restriction.  It changes the
coordinate origin of an already certified point polynomial without re-running
the engineering model.  All coefficient-rounding error introduced by the
coordinate transform is rigorously returned as an interval remainder.
"""

from __future__ import annotations

from itertools import product
from math import comb
import numpy as np

from ..trusted import Interval, point_interval
from .algebra import clean_poly
from .ranging import monomial_range_symmetric


def _beta_indices(alpha):
    return product(*(range(int(a) + 1) for a in alpha))


def rigorous_recenter_point_polynomial(coeff, shift, new_h, backend, *, shift_intervals=None):
    """Recenter ``P(q)`` under ``q = r + shift``.

    Parameters
    ----------
    coeff:
        Mapping from multi-index ``alpha`` to stored binary64 coefficient.
    shift:
        Binary64 proposal vector ``s_hat`` for the exact coordinate shift.
    shift_intervals:
        Optional rigorous intervals containing the exact real shift between the
        two stored binary64 centers.  When omitted the proposal shift is treated
        as exact.
    new_h:
        Nonnegative half-widths of the new centered coordinates ``r``.
    backend:
        Trusted binary64 backend.

    Returns
    -------
    new_coeff, extra_remainder
        ``new_coeff`` is a point-coefficient polynomial proposal in ``r`` and
        ``extra_remainder`` rigorously encloses every exact-real coefficient
        transform error caused by binary64 proposal arithmetic.
    """
    shift = np.asarray(shift, dtype=float)
    new_h = np.asarray(new_h, dtype=float)
    m = len(shift)
    if shift_intervals is None:
        shift_intervals = [point_interval(float(x)) for x in shift]
    else:
        shift_intervals = list(shift_intervals)
        if len(shift_intervals) != m:
            raise ValueError("shift_intervals must match the shift dimension.")
    if len(new_h) != m:
        raise ValueError("shift and new_h must have the same dimension.")
    if np.any(~np.isfinite(shift)) or np.any(~np.isfinite(new_h)) or np.any(new_h < 0.0):
        raise ValueError("Polynomial recentering requires finite shifts and nonnegative half-widths.")

    exact = {}
    proposal = {}

    # Deterministic alpha order makes cache-derived results reproducible.
    for alpha in sorted(coeff):
        alpha = tuple(int(a) for a in alpha)
        if len(alpha) != m:
            raise ValueError("Polynomial coefficient dimension does not match shift dimension.")
        c = float(coeff[alpha])
        for beta_raw in _beta_indices(alpha):
            beta = tuple(int(b) for b in beta_raw)

            # Exact-real contribution c * prod_i C(alpha_i,beta_i) s_i^(alpha_i-beta_i),
            # enclosed solely through trusted binary64 interval operations.
            term_I = point_interval(c)
            term_p = c
            for i, (ai, bi) in enumerate(zip(alpha, beta)):
                choose = comb(ai, bi)
                if choose > 2**53:
                    raise OverflowError(
                        "Polynomial recentering encountered a binomial coefficient larger "
                        "than the exact binary64 integer range."
                    )
                if choose != 1:
                    term_I = backend.mul(term_I, point_interval(float(choose)))
                    term_p = float(term_p * float(choose))
                power = ai - bi
                for _ in range(power):
                    term_I = backend.mul(term_I, shift_intervals[i])
                    term_p = float(term_p * float(shift[i]))

            if beta in exact:
                exact[beta] = backend.add(exact[beta], term_I)
                proposal[beta] = float(proposal[beta] + term_p)
            else:
                exact[beta] = term_I
                proposal[beta] = float(term_p)
            if not np.isfinite(proposal[beta]):
                raise FloatingPointError("Nonfinite polynomial coefficient during recentering.")

    out = clean_poly(proposal)
    remainder = Interval(0.0, 0.0)
    for beta, exact_I in exact.items():
        p = float(proposal.get(beta, 0.0))
        coeff_error = backend.sub(exact_I, point_interval(p))
        if coeff_error.lo == coeff_error.hi == 0.0:
            continue
        monomial_I = monomial_range_symmetric(beta, new_h, backend)
        remainder = backend.add(remainder, backend.mul(coeff_error, monomial_I))

    return out, remainder


__all__ = ["rigorous_recenter_point_polynomial"]
