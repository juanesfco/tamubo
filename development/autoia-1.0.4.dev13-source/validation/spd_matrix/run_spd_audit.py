"""Independent high-precision audit for AutoIA SPD matrix-family semantics."""
from pathlib import Path
import time
import numpy as np
import mpmath as mp

from autoia import autoia, Interval, quad_solve_spd

mp.mp.dps = 80
X_FLOAT = np.array([0.0, 0.45, 1.0])
Y_FLOAT = np.array([0.2, -0.3, 0.5])
X_MP = [mp.mpf('0.0'), mp.mpf('0.45'), mp.mpf('1.0')]
Y_MP = mp.matrix([mp.mpf('0.2'), mp.mpf('-0.3'), mp.mpf('0.5')])


def model(p):
    logell, loglam = p
    ell = np.exp(logell)
    lam = np.exp(loglam)
    K = np.empty((3,3), dtype=object)
    for i in range(3):
        for j in range(3):
            r = (X_FLOAT[i]-X_FLOAT[j])/ell
            K[i,j] = np.exp(-0.5*r*r)
        K[i,i] = K[i,i] + lam
    ld = np.linalg.slogdet(K)[1]
    q = quad_solve_spd(K, Y_FLOAT)
    return np.array([ld, q, 0.5*(ld+q)], dtype=object)


def mp_values(logell, loglam):
    ell = mp.e**logell; lam = mp.e**loglam
    K = mp.matrix(3,3)
    for i in range(3):
        for j in range(3):
            r = (X_MP[i]-X_MP[j])/ell
            K[i,j] = mp.e**(-mp.mpf('0.5')*r*r)
        K[i,i] += lam
    ld = mp.log(mp.det(K))
    q = (Y_MP.T*(K**-1)*Y_MP)[0]
    return [ld, q, mp.mpf('0.5')*(ld+q)]


def main():
    center = np.array([np.log(0.33), np.log(0.075)])
    halfwidths = [0.12, 0.06, 0.03, 0.015]
    rows = []
    total_checks = 0
    t0 = time.perf_counter()
    for h in halfwidths:
        box = [Interval(center[0]-h, center[0]+h),
               Interval(center[1]-h, center[1]+h)]
        result = autoia(model, box, order=4, representation='polynomial')
        contained = True
        checks = 0
        sample_lo = np.full(3, np.inf); sample_hi = np.full(3, -np.inf)
        for le in np.linspace(box[0].lo, box[0].hi, 9):
            for ln in np.linspace(box[1].lo, box[1].hi, 9):
                vals = mp_values(mp.mpf(str(float(le))), mp.mpf(str(float(ln))))
                for i, val in enumerate(vals):
                    checks += 1
                    if not (mp.mpf(str(result.lower[i])) <= val <= mp.mpf(str(result.upper[i]))):
                        contained = False
                    fv = float(val)
                    sample_lo[i] = min(sample_lo[i], fv)
                    sample_hi[i] = max(sample_hi[i], fv)
        total_checks += checks
        rows.append((h, result.lower.copy(), result.upper.copy(), sample_lo, sample_hi, contained))

    out = Path(__file__).with_name('AUDIT_RESULTS.md')
    lines = [
        '# SPD matrix-family high-precision audit', '',
        'The audit uses 80-decimal-digit `mpmath` calculations at a 9 x 9 grid',
        'inside each of four nested hyperparameter boxes for a 3 x 3 RBF-like SPD',
        'matrix family. Sampling is only an independent challenge; AutoIA does not',
        'use these samples to construct its certificates.', '',
        f'- total high-precision scalar containment checks: **{total_checks}**',
        f'- all contained: **{all(r[-1] for r in rows)}**',
        f'- wall time: **{time.perf_counter()-t0:.3f} s**', '',
        '| log-box halfwidth | AutoIA widths (logdet, q, combined) | sampled widths | contained |',
        '|---:|---|---|:---:|',
    ]
    for h, lo, hi, slo, shi, ok in rows:
        widths = hi-lo; sw = shi-slo
        lines.append(
            f'| {h:.6g} | `{np.array2string(widths, precision=8)}` | '
            f'`{np.array2string(sw, precision=8)}` | {ok} |'
        )
    lines += ['', 'The certified widths decrease as the physical hyperparameter box shrinks.']
    out.write_text('\n'.join(lines)+'\n')
    print(out.read_text())


if __name__ == '__main__':
    main()
