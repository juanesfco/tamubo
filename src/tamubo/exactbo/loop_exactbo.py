from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Callable

from .partition import Box
from .loop_partition import PartitionMaxEISearch

Array = np.ndarray

@dataclass
class BOResult:
    X: Array
    y: Array
    x_best: Array
    y_best: float

class ExactBOLoop:
    """
    Outer exact BO loop:
      1) fit model on (X,y)
      2) run partition-based max-EI search to propose x_next
      3) evaluate oracle at x_next
      4) repeat
    """

    def __init__(self, model, bounds: Array, precision: Array | float | None = None, verbose: bool = False):
        self.model = model
        self.init_box = Box(bounds, True)
        if not precision:
            self.precision = 0.1*self.init_box.width
        elif type(precision) == Array:
            self.precision = precision
        elif type(precision) == float:
            self.precision = precision*self.init_box.width
        else:
            raise TypeError("Precision must be Array, float or None")
        
        self.verbose = verbose
        self.grid = self.create_grid()
        self._oracle: Callable[[Array], Array] | None = None

    def create_grid(self) -> Array:
        """
        Create a dense grid over self.domain with per-dim steps in self.precision.

        Returns
        -------
        X : (k, d) ndarray
            All grid points, row-major (np.meshgrid(..., indexing="ij") then raveled).
        """
        d = self.init_box.dim
        axes = []
        for i in range(d):
            lo, hi = self.init_box.bounds[i, 0], self.init_box.bounds[i, 1]
            step = self.precision[i]
            if step <= 0:
                raise ValueError(f"precision[{i}] must be > 0, got {step}")

            # arange may miss the endpoint due to float, so pad a step
            arr = np.arange(lo, hi + step, step, dtype=float)

            # if we overshot slightly, clamp last point to hi
            if arr[-1] != hi and arr.size > 1:
                arr[-1] = hi

            axes.append(arr)

        # build meshgrid
        grids = np.meshgrid(*axes, indexing="ij")
        # each grid is shape (n1, n2, ..., nd); stack and reshape
        stacked = np.stack([g for g in grids], axis=-1)   # (..., d)
        X = stacked.reshape(-1, d)                        # (k, d)
        return X

    def set_oracle(self, fn: Callable[[Array], Array]):
        self._oracle = fn

    def _evaluate_oracle(self, x: Array) -> Array:
        if self._oracle is None:
            raise NotImplementedError("No oracle set. Call `set_oracle(fn)` first.")
        return self._oracle(x)

    def run(self, X0: Array, y0: Array, budget: int) -> BOResult:
        X, y = X0.copy(), y0.copy()
        for i in range(int(budget)):
            if self.verbose:
                print("ExactBO iteration: ", i)
            self.model.fit(X, y.ravel())

            search = PartitionMaxEISearch(self.model, self.init_box, self.grid, self.precision, self.verbose)
            x_next, _ = search.run()
            if x_next is None:
                break

            y_next = self._evaluate_oracle(x_next.reshape((1,self.init_box.dim)))
            X = np.vstack([X, x_next])
            y = np.concatenate([y, y_next])

        idx = int(np.argmin(y))
        return BOResult(X=X, y=y, x_best=X[idx:idx+1], y_best=float(y[idx]))
