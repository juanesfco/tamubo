"""22 — Parameter-dependent SPD matrix families.

AutoIA now has general validated semantics for symmetric positive-definite (SPD)
matrix families whose entries share the same physical uncertain variables.  The
example exercises three common quantities:

    log(det(A(p)))
    det(A(p))
    b(p)^T A(p)^{-1} b(p)

The proof uses a constant numerical preconditioner only as a proposal.  The
actual SPD, log-determinant, and quadratic-form statements are independently
verified by AutoIA's trusted CPU binary64 backend.
"""
import numpy as np
from autoia import Interval, autoia, quad_solve_spd


def model(p):
    x, y = p
    off = 0.10*x - 0.04*y
    A = np.array([
        [2.0 + 0.15*x*x + 0.05*y, off],
        [off, 1.5 + 0.08*y*y - 0.03*x],
    ], dtype=object)
    b = np.array([1.0 + 0.2*x, 0.5 - 0.1*y], dtype=object)

    sign, logdet = np.linalg.slogdet(A)
    det = np.linalg.det(A)
    q = quad_solve_spd(A, b)
    return np.array([sign, logdet, det, q, 0.5*(logdet + q)], dtype=object)


if __name__ == "__main__":
    box = [Interval(-0.25, 0.25), Interval(-0.20, 0.30)]
    result = autoia(model, box)
    print(result)
    print("machine rigorous:", result.machine_rigorous)
    print("outputs: sign, logdet, det, inverse quadratic form, combined objective")
    print("lower:", result.lower)
    print("upper:", result.upper)

    # Sampling is diagnostic only and is not part of the proof.
    sample_min = np.full(5, np.inf)
    sample_max = np.full(5, -np.inf)
    for x in np.linspace(box[0].lo, box[0].hi, 31):
        for y in np.linspace(box[1].lo, box[1].hi, 31):
            off = 0.10*x - 0.04*y
            A = np.array([
                [2.0 + 0.15*x*x + 0.05*y, off],
                [off, 1.5 + 0.08*y*y - 0.03*x],
            ])
            b = np.array([1.0 + 0.2*x, 0.5 - 0.1*y])
            sign, ld = np.linalg.slogdet(A)
            det = np.linalg.det(A)
            q = float(b @ np.linalg.solve(A, b))
            vals = np.array([sign, ld, det, q, 0.5*(ld + q)])
            sample_min = np.minimum(sample_min, vals)
            sample_max = np.maximum(sample_max, vals)
    print("sample lower (diagnostic only):", sample_min)
    print("sample upper (diagnostic only):", sample_max)
