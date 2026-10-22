"""Independent high-precision audit for AutoIA implicit solve semantics."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy.optimize import root_scalar, fsolve

from autoia import autoia, Interval

mp.mp.dps = 80


def colebrook_resid_np(f, Re, rr):
    if f <= 0:
        return np.nan
    term = rr / 3.7 + 2.51 / (Re * np.sqrt(f))
    return 1.0 / np.sqrt(f) + 2.0 * np.log10(term)


def colebrook_model(p):
    return root_scalar(
        colebrook_resid_np, bracket=[0.0008, 0.1], args=(p[0], p[1])
    ).root


def colebrook_mp(Re, rr):
    Re = mp.mpf(Re)
    rr = mp.mpf(rr)
    lo = mp.mpf("0.0008")
    hi = mp.mpf("0.1")

    def F(f):
        return 1 / mp.sqrt(f) + 2 * mp.log10(rr / mp.mpf("3.7") + mp.mpf("2.51") / (Re * mp.sqrt(f)))

    flo, fhi = F(lo), F(hi)
    assert flo * fhi < 0
    for _ in range(300):
        mid = (lo + hi) / 2
        fm = F(mid)
        if fm == 0:
            return mid
        if flo * fm <= 0:
            hi = mid
            fhi = fm
        else:
            lo = mid
            flo = fm
    return (lo + hi) / 2


def system_np(y, a, b):
    return np.array([
        y[0] + 0.2 * y[1] - a,
        0.1 * y[0] + y[1] + 0.05 * y[0] * y[1] - b,
    ])


def fsolve_model(p):
    return fsolve(system_np, [1.0, 1.5], args=(p[0], p[1]))


def system_mp(a, b):
    a = mp.mpf(a)
    b = mp.mpf(b)
    def f1(x, y): return x + mp.mpf("0.2") * y - a
    def f2(x, y): return mp.mpf("0.1") * x + y + mp.mpf("0.05") * x * y - b
    x, y = mp.findroot((f1, f2), (mp.mpf("0.7"), mp.mpf("1.4")), tol=mp.mpf("1e-70"), maxsteps=100)
    return x, y


def main():
    rng = np.random.default_rng(20260830)
    root_checks = 0
    fsolve_checks = 0
    root_boxes = 0
    fsolve_boxes = 0

    # Include the user-supplied acceptance region plus randomized nearby boxes.
    root_specs = [(90000.0, 110000.0, 0.0008, 0.0012)]
    for _ in range(3):
        Re_c = rng.uniform(6e4, 2e5)
        Re_h = rng.uniform(2e3, 1.5e4)
        rr_c = rng.uniform(2e-4, 3e-3)
        rr_h = rng.uniform(2e-5, min(4e-4, 0.7 * rr_c))
        root_specs.append((Re_c-Re_h, Re_c+Re_h, rr_c-rr_h, rr_c+rr_h))

    root_widths = []
    for Re_lo, Re_hi, rr_lo, rr_hi in root_specs:
        result = autoia(colebrook_model, [Interval(Re_lo, Re_hi), Interval(rr_lo, rr_hi)])
        assert result.machine_rigorous
        root_boxes += 1
        root_widths.append(float(result.width[0]))
        for _ in range(12):
            Re = rng.uniform(Re_lo, Re_hi)
            rr = rng.uniform(rr_lo, rr_hi)
            exact = colebrook_mp(str(Re), str(rr))
            lo = mp.mpf(str(result.lower[0]))
            hi = mp.mpf(str(result.upper[0]))
            assert lo <= exact <= hi, (Re, rr, exact, lo, hi)
            root_checks += 1

    fsolve_specs = [(0.9, 1.1, 1.4, 1.6)]
    for _ in range(3):
        a_c = rng.uniform(0.8, 1.2)
        b_c = rng.uniform(1.3, 1.7)
        ah = rng.uniform(0.01, 0.08)
        bh = rng.uniform(0.01, 0.08)
        fsolve_specs.append((a_c-ah, a_c+ah, b_c-bh, b_c+bh))

    fsolve_widths = []
    for a_lo, a_hi, b_lo, b_hi in fsolve_specs:
        result = autoia(fsolve_model, [Interval(a_lo, a_hi), Interval(b_lo, b_hi)])
        assert result.machine_rigorous
        fsolve_boxes += 1
        fsolve_widths.append([float(w) for w in result.width])
        for _ in range(12):
            a = rng.uniform(a_lo, a_hi)
            b = rng.uniform(b_lo, b_hi)
            x, y = system_mp(str(a), str(b))
            vals = [x, y]
            for j, exact in enumerate(vals):
                lo = mp.mpf(str(result.lower[j]))
                hi = mp.mpf(str(result.upper[j]))
                assert lo <= exact <= hi, (a, b, j, exact, lo, hi)
                fsolve_checks += 1

    summary = {
        "mp_dps": int(mp.mp.dps),
        "root_scalar_boxes": root_boxes,
        "root_scalar_high_precision_point_checks": root_checks,
        "fsolve_boxes": fsolve_boxes,
        "fsolve_high_precision_component_checks": fsolve_checks,
        "root_scalar_max_width": max(root_widths),
        "fsolve_max_component_width": max(max(v) for v in fsolve_widths),
        "status": "PASS",
    }
    out = Path(__file__).resolve().parents[2] / "STEP20J_IMPLICIT_SOLVE_AUDIT.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
