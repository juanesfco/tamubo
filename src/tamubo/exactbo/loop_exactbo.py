from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Callable
from copy import deepcopy
import time
from sklearn.base import clone

from .partition import Box
from .loop_partition import PartitionMaxEISearch
from .ei import expected_improvement
from .plots import plot_iterations, plot_iterations_2d

Array = np.ndarray

@dataclass
class BOResult:
    X     : Array
    y     : Array
    X_opt : Array | None = None
    y_min : float | None = None
    
    def __post_init__(self):
        idx = int(np.argmin(self.y))
        self.X_opt = self.X[idx]
        self.y_min = float(self.y[idx])

class ExactBOLoop:
    """
    Outer exact BO loop:
      1) fit model on (X,y)
      2) run partition-based max-EI search to propose x_next
      3) evaluate oracle at x_next
      4) repeat
    """

    def __init__(self, model, bounds: Array, precision: Array | float | None = None, log: dict | bool = False):
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
        
        if log:
            self.log = {"ebo_log": True}
        else:
            self.log = log
        
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
    
    def estimate_opt(self):
        d = self.init_box.dim
        axes = []
        for i in range(d):
            lo, hi = self.init_box.bounds[i, 0], self.init_box.bounds[i, 1]
            arr = np.linspace(lo, hi, 1000, dtype=float)
            axes.append(arr)

        # build meshgrid
        grids = np.meshgrid(*axes, indexing="ij")
        # each grid is shape (n1, n2, ..., nd); stack and reshape
        stacked = np.stack([g for g in grids], axis=-1)   # (..., d)
        X = stacked.reshape(-1, d)                        # (k, d)
        y = self._evaluate_oracle(X)
        i_min = y.argmin()
        x_min = X[i_min]
        y_min = y[i_min]
        return BOResult(x_min.reshape(1,d), np.array([y_min]))

    def run(self, X0: Array, y0: Array, budget: int, max_splits: int = 100, split_type: str = "full") -> BOResult:
        if self.log:
            self.log["start"] = {"oracle": self._oracle, "domain": self.init_box.bounds, "precision": self.precision}
            Xc, yc = X0.copy(), y0.copy()
            modelc = clone(self.model)
        X, y = X0.copy(), y0.copy()
        for i in range(int(budget)):
            self.model.fit(X, y.ravel())
            
            if self.log:
                self.log[f"ebo_it{i}"] = {"start": BOResult(X, y), "model": deepcopy(self.model)}
            
            if self.log:
                modelc.fit(Xc, yc.ravel())
                boStartTime = time.perf_counter()
                ei_xx = expected_improvement(self.grid, modelc)
                best_i_ei = np.argmax(ei_xx)
                best_x_ei = self.grid[best_i_ei]
                boEndTime = time.perf_counter()
                y_next_ei = self._evaluate_oracle(best_x_ei.reshape((1,self.init_box.dim)))
                Xc = np.vstack([Xc, best_x_ei])
                yc = np.concatenate([yc, y_next_ei])
                eboStartTime = time.perf_counter()
                search = PartitionMaxEISearch(self.model, self.init_box, self.grid, self.precision, self.log)
                x_next, _ = search.run(max_splits, split_type)
                eboEndTime = time.perf_counter()
                self.log[f"ebo_it{i}"]["compare"] = {"timeBO":boEndTime-boStartTime, "timeEBO":eboEndTime-eboStartTime, "best_x_BO": best_x_ei, "best_x_EBO": x_next}

            else:
                search = PartitionMaxEISearch(self.model, self.init_box, self.grid, self.precision, self.log)
                x_next, _ = search.run(max_splits, split_type)
                
            if x_next is None:
                break

            y_next = self._evaluate_oracle(x_next.reshape((1,self.init_box.dim)))
            X = np.vstack([X, x_next])
            y = np.concatenate([y, y_next])

        res = BOResult(X=X, y=y)
        if self.log:
            res_ei = BOResult(X=Xc, y=yc)
            self.log["result"] = {"EBO": res, "BO": res_ei, "Opt": self.estimate_opt()}
        return res
    
    def plot(self, path: str | None = None):
        if self.log:
            d = self.init_box.dim
            if d > 2:
                raise ValueError("Dimension is too high to plot.")
            elif d == 2:
                plot_iterations_2d(self.log, path)
            else:
                plot_iterations(self.log, path)
        else:
            raise AttributeError("Create ExactBOLoop with log=True and then ExactBOLoop.run() before plotting.")