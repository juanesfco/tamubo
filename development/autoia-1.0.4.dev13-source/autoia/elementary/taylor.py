"""Generic Taylor data for validated elementary-function composition.

This module is representation-agnostic.  It provides rigorous interval
coefficients for a Taylor expansion about one stored binary64 center and a
rigorous Lagrange-remainder radius over a real interval.  The actual polynomial
composition is performed by the enclosure layer so coefficient uncertainty,
input remainder, and polynomial truncation are handled by the existing generic
arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..trusted import (
    Interval,
    point_interval,
    integer_interval,
    interval_neg,
    abs_upper_interval,
    sqrt_point_interval,
    exp_point_interval,
    exp_interval,
    log_point_interval,
    ln10_interval,
    sin_cos_interval,
    sinh_point_interval,
    sinh_interval,
    cosh_point_interval,
    cosh_interval,
    tanh_point_interval,
    tanh_interval,
    arctan_point_interval,
    log1p_point_interval,
    erf_point_interval,
    real_power_interval,
    pi_interval,
    sqrt_interval,
    standard_normal_pdf_point_interval,
    standard_normal_pdf_interval,
    standard_normal_cdf_point_interval,
)


SUPPORTED_ELEMENTARY_FUNCTIONS = (
    "sqrt", "exp", "log", "log10", "log1p",
    "sin", "cos", "sinh", "cosh", "tanh", "arctan",
    "erf", "normal_pdf", "normal_cdf",
)


@dataclass(frozen=True)
class TaylorData:
    name: str
    center: float
    domain: Interval
    coefficients: tuple[Interval, ...]
    remainder_radius: float


def _pow_nonnegative(x: Interval, n: int, backend) -> Interval:
    if n < 0:
        raise ValueError("Power must be nonnegative.")
    out = point_interval(1.0)
    base = x
    k = int(n)
    while k:
        if k & 1:
            out = backend.mul(out, base)
        k >>= 1
        if k:
            base = backend.mul(base, base)
    return out


def _positive_integer(n: int) -> Interval:
    return integer_interval(int(n))


def _abs_interval(x: Interval) -> Interval:
    return Interval(0.0, abs_upper_interval(x))


def _coefficient_exp(center: float, degree: int, backend) -> list[Interval]:
    coeff = [exp_point_interval(center, backend)]
    for n in range(1, degree + 1):
        coeff.append(backend.div(coeff[-1], _positive_integer(n)))
    return coeff


def _coefficient_sin_cos(center: float, degree: int, backend, name: str) -> list[Interval]:
    s, c = sin_cos_interval(point_interval(center), backend)
    if name == "sin":
        a0, a1 = s, c
    else:
        a0, a1 = c, interval_neg(s)
    if degree == 0:
        return [a0]
    coeff = [a0, a1]
    # Both sin and cos coefficients satisfy a_n = -a_{n-2}/(n(n-1)).
    for n in range(2, degree + 1):
        denom = _positive_integer(n * (n - 1))
        coeff.append(interval_neg(backend.div(coeff[n - 2], denom)))
    return coeff


def _coefficient_log(center: float, degree: int, backend) -> list[Interval]:
    if center <= 0.0:
        raise ValueError("log Taylor center must be positive")
    coeff = [log_point_interval(center, backend)]
    if degree == 0:
        return coeff
    c = point_interval(center)
    coeff.append(backend.div(point_interval(1.0), c))
    # a_n = -a_{n-1} * (n-1)/(n*c)
    for n in range(2, degree + 1):
        num = backend.mul(coeff[-1], _positive_integer(n - 1))
        den = backend.mul(_positive_integer(n), c)
        coeff.append(interval_neg(backend.div(num, den)))
    return coeff


def _coefficient_log10(center: float, degree: int, backend) -> list[Interval]:
    """Taylor coefficients of ``log10`` via rigorous scaling of ``ln``."""
    scale = ln10_interval(backend)
    return [backend.div(a, scale) for a in _coefficient_log(center, degree, backend)]


def _coefficient_log1p(center: float, degree: int, backend) -> list[Interval]:
    if center <= -1.0:
        raise ValueError("log1p Taylor center must be greater than -1")
    shifted = backend.add(point_interval(1.0), point_interval(center))
    if shifted.lo <= 0.0:
        raise ValueError("log1p shifted Taylor center must be positive")
    coeff = [log1p_point_interval(center, backend)]
    if degree == 0:
        return coeff
    coeff.append(backend.div(point_interval(1.0), shifted))
    for n in range(2, degree + 1):
        num = backend.mul(coeff[-1], _positive_integer(n - 1))
        den = backend.mul(_positive_integer(n), shifted)
        coeff.append(interval_neg(backend.div(num, den)))
    return coeff


def _coefficient_sinh_cosh(center: float, degree: int, backend, name: str) -> list[Interval]:
    sh = sinh_point_interval(center, backend)
    ch = cosh_point_interval(center, backend)
    a0, a1 = (sh, ch) if name == "sinh" else (ch, sh)
    if degree == 0:
        return [a0]
    coeff = [a0, a1]
    for n in range(2, degree + 1):
        coeff.append(backend.div(coeff[n - 2], _positive_integer(n * (n - 1))))
    return coeff


def _coefficient_tanh(center: float, degree: int, backend) -> list[Interval]:
    coeff = [tanh_point_interval(center, backend)]
    for n in range(degree):
        conv = point_interval(0.0)
        for k in range(n + 1):
            conv = backend.add(conv, backend.mul(coeff[k], coeff[n - k]))
        rhs = backend.sub(point_interval(1.0), conv) if n == 0 else interval_neg(conv)
        coeff.append(backend.div(rhs, _positive_integer(n + 1)))
    return coeff[: degree + 1]


def _coefficient_arctan(center: float, degree: int, backend) -> list[Interval]:
    coeff = [arctan_point_interval(center, backend)]
    if degree == 0:
        return coeff
    c = point_interval(center)
    q0 = backend.add(point_interval(1.0), real_power_interval(c, 2.0, backend))
    q1 = backend.mul(point_interval(2.0), c)
    coeff.append(backend.div(point_interval(1.0), q0))
    for n in range(1, degree):
        term1 = backend.mul(q1, backend.mul(_positive_integer(n), coeff[n]))
        term2 = point_interval(0.0) if n == 1 else backend.mul(_positive_integer(n - 1), coeff[n - 1])
        num = backend.add(term1, term2)
        den = backend.mul(q0, _positive_integer(n + 1))
        coeff.append(interval_neg(backend.div(num, den)))
    return coeff


def _coefficient_sqrt(center: float, degree: int, backend) -> list[Interval]:
    if center <= 0.0:
        raise ValueError("sqrt Taylor center must be positive")
    coeff = [sqrt_point_interval(center, backend)]
    c = point_interval(center)
    # a_n = a_{n-1} * (3-2n)/(2*n*c)
    for n in range(1, degree + 1):
        factor_num = integer_interval(3 - 2 * n)
        factor_den = backend.mul(_positive_integer(2 * n), c)
        coeff.append(backend.div(backend.mul(coeff[-1], factor_num), factor_den))
    return coeff



def _two_over_sqrt_pi(backend) -> Interval:
    sqrt_pi = sqrt_interval(pi_interval(backend), backend)
    return backend.div(point_interval(2.0), sqrt_pi)


def _coefficient_erf(center: float, degree: int, backend) -> list[Interval]:
    a0 = erf_point_interval(center, backend)
    if degree == 0:
        return [a0]
    c = point_interval(center)
    c2 = backend.mul(c, c)
    a1 = backend.mul(
        _two_over_sqrt_pi(backend),
        exp_interval(interval_neg(c2), backend),
    )
    coeff = [a0, a1]
    for n in range(0, degree - 1):
        # (n+2)(n+1)a_{n+2} + 2c(n+1)a_{n+1} + 2n a_n = 0.
        term1 = backend.mul(
            backend.mul(point_interval(2.0), c),
            backend.mul(_positive_integer(n + 1), coeff[n + 1]),
        )
        term2 = backend.mul(_positive_integer(2 * n), coeff[n]) if n else point_interval(0.0)
        num = backend.add(term1, term2)
        den = _positive_integer((n + 2) * (n + 1))
        coeff.append(interval_neg(backend.div(num, den)))
    return coeff[: degree + 1]


def _coefficient_normal_pdf(center: float, degree: int, backend) -> list[Interval]:
    a0 = standard_normal_pdf_point_interval(center, backend)
    coeff = [a0]
    if degree == 0:
        return coeff
    c = point_interval(center)
    coeff.append(interval_neg(backend.mul(c, a0)))
    for n in range(1, degree):
        num = backend.add(backend.mul(c, coeff[n]), coeff[n - 1])
        coeff.append(interval_neg(backend.div(num, _positive_integer(n + 1))))
    return coeff


def _coefficient_normal_cdf(center: float, degree: int, backend) -> list[Interval]:
    out = [standard_normal_cdf_point_interval(center, backend)]
    if degree == 0:
        return out
    pdf = _coefficient_normal_pdf(center, degree - 1, backend)
    for n in range(1, degree + 1):
        out.append(backend.div(pdf[n - 1], _positive_integer(n)))
    return out


def _erf_derivative_coefficient_range(domain: Interval, n: int, backend) -> Interval:
    """Range of ``erf^(n)(x)/n!`` over ``domain`` for n>=1."""
    n = int(n)
    if n < 1:
        raise ValueError("erf derivative coefficient index must be >= 1")
    x2 = real_power_interval(domain, 2.0, backend)
    a1 = backend.mul(
        _two_over_sqrt_pi(backend),
        exp_interval(interval_neg(x2), backend),
    )
    if n == 1:
        return a1
    prev = None
    cur = a1
    for j in range(0, n - 1):
        # Compute a_{j+2}; a_0 is not needed because its coefficient is 2j=0 at j=0.
        term1 = backend.mul(
            backend.mul(point_interval(2.0), domain),
            backend.mul(_positive_integer(j + 1), cur),
        )
        term2 = point_interval(0.0) if j == 0 else backend.mul(_positive_integer(2 * j), prev)
        nxt = interval_neg(backend.div(backend.add(term1, term2), _positive_integer((j + 2) * (j + 1))))
        prev, cur = cur, nxt
    return cur


def _normal_pdf_derivative_coefficient_range(domain: Interval, n: int, backend) -> Interval:
    """Range of ``phi^(n)(x)/n!`` for the standard normal PDF."""
    n = int(n)
    b0 = standard_normal_pdf_interval(domain, backend)
    if n == 0:
        return b0
    prev = b0
    cur = interval_neg(backend.mul(domain, b0))
    if n == 1:
        return cur
    for j in range(1, n):
        nxt = interval_neg(backend.div(backend.add(backend.mul(domain, cur), prev), _positive_integer(j + 1)))
        prev, cur = cur, nxt
    return cur

def _tanh_derivative_coefficient_range(domain: Interval, n: int, backend) -> Interval:
    """Range of ``tanh^(n)(x)/n!`` over ``domain``."""
    n = int(n)
    coeff = [tanh_interval(domain, backend)]
    if n == 0:
        return coeff[0]
    for j in range(n):
        conv = point_interval(0.0)
        for k in range(j + 1):
            conv = backend.add(conv, backend.mul(coeff[k], coeff[j - k]))
        rhs = backend.sub(point_interval(1.0), conv) if j == 0 else interval_neg(conv)
        coeff.append(backend.div(rhs, _positive_integer(j + 1)))
    return coeff[n]


def _arctan_derivative_coefficient_range(domain: Interval, n: int, backend) -> Interval:
    """Range of ``atan^(n)(x)/n!`` over ``domain`` for ``n >= 1``."""
    n = int(n)
    if n < 1:
        raise ValueError("arctan derivative coefficient index must be >= 1")
    q0 = backend.add(point_interval(1.0), real_power_interval(domain, 2.0, backend))
    q1 = backend.mul(point_interval(2.0), domain)
    a = [None, backend.div(point_interval(1.0), q0)]
    if n == 1:
        return a[1]
    for j in range(1, n):
        term1 = backend.mul(q1, backend.mul(_positive_integer(j), a[j]))
        term2 = point_interval(0.0) if j == 1 else backend.mul(_positive_integer(j - 1), a[j - 1])
        den = backend.mul(q0, _positive_integer(j + 1))
        a.append(interval_neg(backend.div(backend.add(term1, term2), den)))
    return a[n]


def taylor_coefficients(name: str, center: float, degree: int, backend) -> tuple[Interval, ...]:
    """Rigorous intervals containing ``f^(n)(center)/n!`` for n=0..degree."""
    name = str(name)
    if name not in SUPPORTED_ELEMENTARY_FUNCTIONS:
        raise NotImplementedError(f"Unsupported elementary function {name!r}.")
    degree = int(degree)
    if degree < 0:
        raise ValueError("Taylor degree must be nonnegative.")
    center = float(center)
    if name == "exp":
        out = _coefficient_exp(center, degree, backend)
    elif name in {"sin", "cos"}:
        out = _coefficient_sin_cos(center, degree, backend, name)
    elif name == "log":
        out = _coefficient_log(center, degree, backend)
    elif name == "log10":
        out = _coefficient_log10(center, degree, backend)
    elif name == "log1p":
        out = _coefficient_log1p(center, degree, backend)
    elif name == "sqrt":
        out = _coefficient_sqrt(center, degree, backend)
    elif name in {"sinh", "cosh"}:
        out = _coefficient_sinh_cosh(center, degree, backend, name)
    elif name == "tanh":
        out = _coefficient_tanh(center, degree, backend)
    elif name == "arctan":
        out = _coefficient_arctan(center, degree, backend)
    elif name == "erf":
        out = _coefficient_erf(center, degree, backend)
    elif name == "normal_pdf":
        out = _coefficient_normal_pdf(center, degree, backend)
    else:
        out = _coefficient_normal_cdf(center, degree, backend)
    return tuple(out)


def _sqrt_binomial_coefficient_abs(n: int, backend) -> Interval:
    """Rigorous ``abs(binomial(1/2,n))`` for n >= 0."""
    coeff = point_interval(1.0)
    for j in range(1, int(n) + 1):
        # (1/2-(j-1))/j = (3-2j)/(2j)
        num = integer_interval(3 - 2 * j)
        den = _positive_integer(2 * j)
        coeff = backend.mul(coeff, backend.div(num, den))
    return _abs_interval(coeff)


def taylor_remainder_radius(
    name: str,
    center: float,
    degree: int,
    domain: Interval,
    delta_abs: float,
    backend,
) -> float:
    """Rigorous Lagrange-remainder radius for a degree-``degree`` expansion.

    ``delta_abs`` must rigorously bound ``|x-center|`` for every x in domain.
    """
    name = str(name)
    degree = int(degree)
    n = degree + 1
    d = point_interval(float(delta_abs))
    d_pow = _pow_nonnegative(d, n, backend)

    if name in {"sin", "cos"}:
        # All real derivatives of sin and cos have absolute value <= 1.
        bound = d_pow
        for k in range(2, n + 1):
            bound = backend.div(bound, _positive_integer(k))
        return float(bound.hi)

    if name == "exp":
        # Every derivative is exp; monotonicity gives max at domain.hi.
        m = exp_interval(domain, backend).hi
        bound = backend.mul(d_pow, point_interval(m))
        for k in range(2, n + 1):
            bound = backend.div(bound, _positive_integer(k))
        return float(bound.hi)

    if name in {"sinh", "cosh"}:
        # Every derivative is sinh or cosh.  Both absolute values are bounded
        # by cosh(A), A=max(|lo|,|hi|).
        A = point_interval(abs_upper_interval(domain))
        M = cosh_interval(A, backend).hi
        bound = backend.mul(d_pow, point_interval(M))
        for k in range(2, n + 1):
            bound = backend.div(bound, _positive_integer(k))
        return float(bound.hi)

    if name == "tanh":
        an = _tanh_derivative_coefficient_range(domain, n, backend)
        return float(backend.mul(_abs_interval(an), d_pow).hi)

    if name == "arctan":
        an = _arctan_derivative_coefficient_range(domain, n, backend)
        return float(backend.mul(_abs_interval(an), d_pow).hi)

    if name == "log1p":
        if domain.lo <= -1.0:
            raise ValueError("log1p remainder requires domain > -1")
        shifted = backend.add(point_interval(1.0), point_interval(domain.lo))
        if shifted.lo <= 0.0:
            raise ValueError("log1p shifted remainder domain must be positive")
        xpow = _pow_nonnegative(shifted, n, backend)
        den = backend.mul(_positive_integer(n), xpow)
        an_abs = backend.div(point_interval(1.0), den)
        return float(backend.mul(an_abs, d_pow).hi)

    if name in {"log", "log10"}:
        if domain.lo <= 0.0:
            raise ValueError(f"{name} remainder requires a positive domain")
        # Around positive center c, log(c+delta)=log(c)+log(1+u),
        # u=delta/c.  For |u|<=a<1, the absolute tail after degree K obeys
        #   sum_{j=n} inf a^j/j <= a^n/(n*(1-a)), n=K+1.
        # This respects the actual radius of analyticity (the singularity at 0)
        # and is dramatically sharper than a global Lagrange derivative bound.
        center = float(center)
        if center > 0.0:
            aI = backend.div(point_interval(float(delta_abs)), point_interval(center))
            if aI.hi < 1.0:
                apow = _pow_nonnegative(Interval(0.0, aI.hi), n, backend)
                den = backend.mul(
                    _positive_integer(n),
                    backend.sub(point_interval(1.0), point_interval(aI.hi)),
                )
                log_bound = backend.div(apow, den)
                if name == "log10":
                    log_bound = backend.div(log_bound, ln10_interval(backend))
                return float(log_bound.hi)
        # Conservative fallback.
        xpow = _pow_nonnegative(point_interval(domain.lo), n, backend)
        den = backend.mul(_positive_integer(n), xpow)
        log_bound = backend.div(d_pow, den)
        if name == "log10":
            log_bound = backend.div(log_bound, ln10_interval(backend))
        return float(log_bound.hi)

    if name == "sqrt":
        if domain.lo <= 0.0:
            raise ValueError("sqrt Taylor remainder requires a strictly positive domain")
        # sqrt(c+delta)=sqrt(c)*(1+u)^(1/2), u=delta/c.
        # The binomial-term ratio after the first term is strictly below |u|.
        # Thus for |u|<=a<1 the tail is bounded by the first omitted term
        # divided by (1-a).
        center = float(center)
        if center > 0.0:
            aI = backend.div(point_interval(float(delta_abs)), point_interval(center))
            if aI.hi < 1.0:
                apow = _pow_nonnegative(Interval(0.0, aI.hi), n, backend)
                binom = _sqrt_binomial_coefficient_abs(n, backend)
                sqrt_c = sqrt_point_interval(center, backend)
                first = backend.mul(backend.mul(sqrt_c, binom), apow)
                geom = backend.div(
                    point_interval(1.0),
                    backend.sub(point_interval(1.0), point_interval(aI.hi)),
                )
                return float(backend.mul(first, geom).hi)
        # Conservative Lagrange fallback.
        an = taylor_coefficients("sqrt", domain.lo, n, backend)[n]
        return float(backend.mul(_abs_interval(an), d_pow).hi)


    if name == "erf":
        an = _erf_derivative_coefficient_range(domain, n, backend)
        return float(backend.mul(_abs_interval(an), d_pow).hi)

    if name == "normal_pdf":
        an = _normal_pdf_derivative_coefficient_range(domain, n, backend)
        return float(backend.mul(_abs_interval(an), d_pow).hi)

    if name == "normal_cdf":
        # Phi^(n)/n! = phi^(n-1)/n! = b_{n-1}/n for n>=1.
        bn1 = _normal_pdf_derivative_coefficient_range(domain, n - 1, backend)
        an = backend.div(bn1, _positive_integer(n))
        return float(backend.mul(_abs_interval(an), d_pow).hi)

    raise NotImplementedError(f"Unsupported elementary function {name!r}.")


def build_taylor_data(
    name: str,
    center: float,
    degree: int,
    domain: Interval,
    delta_abs: float,
    backend,
) -> TaylorData:
    coeff = taylor_coefficients(name, center, degree, backend)
    rem = taylor_remainder_radius(name, center, degree, domain, delta_abs, backend)
    return TaylorData(name, float(center), domain, coeff, rem)


__all__ = [
    "SUPPORTED_ELEMENTARY_FUNCTIONS",
    "TaylorData",
    "taylor_coefficients",
    "taylor_remainder_radius",
    "build_taylor_data",
]
