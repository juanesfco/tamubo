from __future__ import annotations
from dataclasses import dataclass
import numpy as np

Array = np.ndarray

@dataclass
class Bounds:
    lo: float | Array
    hi: float | Array

    def __post_init__(self):
        # 1) coerce to numpy if array
        if (type(self.lo) not in [int, float]) and (type(self.hi) not in [int, float]):
            self.lo = np.asarray(self.lo, dtype=float)
            self.hi = np.asarray(self.hi, dtype=float)
    
    @property
    def asarray(self) -> Array:
        if type(self.lo) == Array and type(self.hi) == Array:
            bounds = np.stack((self.lo,self.hi), axis=1)
            return bounds
        else:
            raise TypeError("self.lo and self.hi must be arrays.")

def prod_bound_scalar(c: float, X: Bounds):
    """Return bounds for c*X, where X is Bounds(lo,hi)."""
    a,b = c*X.lo, c*X.hi
    return Bounds(min(a,b), max(a,b))

def add_bounds(X: Bounds, Y: Bounds):
    """Return bounds for X + Y, where X and Y are Bounds(lo,hi)."""
    return Bounds(X.lo + Y.lo, X.hi + Y.hi)

def sub_bounds(X: Bounds, Y: Bounds):
    """Return bounds for X - Y, where X and Y are Bounds(lo,hi)."""
    return Bounds(X.lo - Y.hi, X.hi - Y.lo)

def prod_bounds(X: Bounds, Y: Bounds):
    """Return bounds for X * Y, where X and Y are Bounds(lo,hi)."""
    possible = [X.lo*Y.lo, X.lo*Y.hi, X.hi*Y.lo, X.hi*Y.hi]
    return Bounds(min(possible), max(possible))

def square_bounds(X: Bounds):
    """Return bounds for X^2, where X is Bounds(lo,hi)."""
    if X.lo <= 0.0 <= X.hi:
        return Bounds(0, max(X.lo*X.lo, X.hi*X.hi))
    else:
        return Bounds(min(X.lo*X.lo, X.hi*X.hi), max(X.lo*X.lo, X.hi*X.hi))
    
def sqrt_bounds(X: Bounds):
    """Return bounds for sqrt(X), where X is Bounds(lo,hi)."""
    return Bounds(X.lo**(0.5), X.hi**(0.5))
    
def forward_solve_bounds(L, k):
    """
    Given lower-triangular L with positive diagonal (Cholesky),
    and componentwise bounds for kernel k = Bounds([lo_i], [hi_i]),
    compute tight interval bounds v = Bounds([lo_i], [hi_i]) for the solution of L v = k.

    Recurrence:
      v_1 = k_1 / L_11
      v_j = (k_j - Σ_{i<j} L_{j,i} v_i) / L_{j,j}
    """
    L = np.asarray(L, float)
    n = L.shape[0]
    v = Bounds(np.zeros(n), np.zeros(n))

    for j in range(n):
        # S_j = Σ_{i<j} L_{j,i} v_i, with each term interval-bounded
        S_j = Bounds(0, 0)
        for i in range(j):
            lij = L[j, i]
            v_i = Bounds(v.lo[i], v.hi[i])
            t = prod_bound_scalar(lij, v_i)
            S_j = add_bounds(S_j, t)

        # Numerator interval: N_j = k_j - S_j
        k_j = Bounds(k.lo[j], k.hi[j])
        N_j = sub_bounds(k_j, S_j)

        # Divide by positive L_{j,j} and edit v
        Ljj = L[j, j]
        v.lo[j] = N_j.lo/Ljj
        v.hi[j] = N_j.hi/Ljj

    return v