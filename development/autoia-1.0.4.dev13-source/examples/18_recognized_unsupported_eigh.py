"""18 — A recognized but deliberately unsupported V1 semantic.

SciPy's symmetric/generalized-symmetric eigensolver is common engineering code.
AutoIA V1 recognizes scipy.linalg.eigh when uncertain matrix entries reach it,
but does not yet claim a validated eigenvalue/eigenvector theorem.  It therefore
fails explicitly closed instead of leaking SciPy's object-array error.
"""
import numpy as np
from autoia import AutoIAError, Interval, autoia

try:
    from scipy.linalg import eigh
except ImportError:
    print("Example 18 skipped: SciPy is not installed.")
    raise SystemExit(0)


def modal_model(p):
    k = p[0]
    K = np.array([[2.0*k, -k], [-k, 2.0*k]])
    M = np.array([[2.0, 0.0], [0.0, 1.0]])
    lam = eigh(K, M, eigvals_only=True)
    return np.sqrt(lam[0]) / (2.0*np.pi)


if __name__ == "__main__":
    try:
        autoia(modal_model, [Interval(90.0, 110.0)])
    except AutoIAError as exc:
        print("Expected fail-closed result:", type(exc).__name__)
        print(exc)
        assert "scipy.linalg.eigh" in str(exc)
        assert "No certificate was produced" in str(exc)
    else:
        raise RuntimeError("Example 18 expected an explicit unsupported-eigh failure.")
