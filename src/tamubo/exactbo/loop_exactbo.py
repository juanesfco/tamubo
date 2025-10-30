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

    def __init__(self, model, bounds: Array):
        self.model = model
        self.bounds = bounds  # (d,2)
        self._oracle: Callable[[Array], Array] | None = None

    def set_oracle(self, fn: Callable[[Array], Array]):
        self._oracle = fn

    def _evaluate_oracle(self, x: Array) -> Array:
        if self._oracle is None:
            raise NotImplementedError("No oracle set. Call `set_oracle(fn)` first.")
        return self._oracle(x)

    def run(self, X0: Array, y0: Array, budget: int) -> BOResult:
        X, y = X0.copy(), y0.copy()
        for _ in range(int(budget)):
            self.model.fit(X, y.ravel())
            f_best = float(y.min())

            search = PartitionMaxEISearch(self.model, f_best, [Box(self.bounds)])
            x_next, _ = search.run(max_iters=200)
            if x_next is None:
                break

            y_next = self._evaluate_oracle(x_next)
            X = np.vstack([X, x_next])
            y = np.vstack([y, y_next])

        idx = int(np.argmin(y))
        return BOResult(X=X, y=y, x_best=X[idx:idx+1], y_best=float(y[idx]))
