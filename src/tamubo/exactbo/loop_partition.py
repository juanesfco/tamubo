from __future__ import annotations
import numpy as np
from typing import List, Tuple

from .partition import Box, split_box, hypermask
from .ei import expected_improvement
from .ei_bounds import ei_bounds_from_mu_sigma_intervals, EIBounds

Array = np.ndarray

class PartitionMaxEISearch:
    """
    Branch-and-bound search for max EI over partitioned boxes WITHOUT updating the GP.

    Assumes `model.predict(X, return_std=True) -> (mu, std)` with (n,1) arrays.
    """

    def __init__(self, model, init_box: Box, precision: Array | float | None = None, grid: Array | None = None):
        self.model = model
        self.domain = init_box.bounds
        self.boxes: List[Box] = [init_box]
        if not precision:
            self.precision = 0.1*init_box.width
        elif precision.dtype == Array:
            self.precision = precision
        else:
            self.precision = precision*init_box.width            
        
        if grid:
            self.grid = grid
        else:
            self.grid = self.create_grid()

        self.max_ei: float = 0.0
        self.best_x: Array | None = None

    def ei_bound(self, box: Box) -> float:
        """Crude but safe: evaluate μ,σ on corners and take min/max."""
        
        return None

    def create_grid(self) -> Array:
        """
        Create a dense grid over self.domain with per-dim steps in self.precision.

        Returns
        -------
        X : (k, d) ndarray
            All grid points, row-major (np.meshgrid(..., indexing="ij") then raveled).
        """
        d = self.domain.shape[0]
        axes = []
        for i in range(d):
            lo, hi = self.domain[i, 0], self.domain[i, 1]
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
    
    def sample_ei_active_boxes(self, percentage: float = 0.01, seed = None) -> float:
        active_boxes_bounds = []
        for box in self.boxes:
            if box.active:
                active_boxes_bounds.append(box.bounds)
        mask_big = hypermask(np.array(active_boxes_bounds), self.grid)

        rng = np.random.default_rng(seed)
        grid_sample = rng.choice(self.grid[mask_big], size=int(round(percentage*mask_big.sum())), replace=False)
        ei_sample = expected_improvement(grid_sample, self.model)
        best_id = np.argmax(ei_sample)
        best_x = grid_sample[best_id]
        max_ei = ei_sample[best_id]
        return best_x, max_ei
    
    def sample_ei_active_boxes_centers(self) -> float:
        active_boxes_centers = []
        for box in self.boxes:
            if box.active:
                active_boxes_centers.append(box.center)
        active_boxes_centers = np.array(active_boxes_centers)
        ei_sample = expected_improvement(active_boxes_centers, self.model)
        best_id = np.argmax(ei_sample)
        best_x = active_boxes_centers[best_id]
        max_ei = ei_sample[best_id]
        return best_x, max_ei
    
    def check_sampled(self, bounds):
        X = self.model.X_train_
        mask = hypermask(bounds, X)
        if np.sum(mask) > 0:
            return True
        else:
            return False

    def run(self, max_iters: int = 10):
        it = 0
        while it < max_iters:
            boxes_it = []
            it += 1
            best_x, max_ei = self.sample_ei_active_boxes()
            if max_ei > self.max_ei:
                self.max_ei = max_ei
                self.best_x = best_x
            for box in self.boxes:
                if not box.active:
                    boxes_it.append(box)
                else:
                    if box.sampled:
                        if self.check_sampled(box.bounds):
                            boxes_it = boxes_it + split_box[box]
                        else:
                            box.sampled = False
                            if self.ei_bound(box) >= self.max_ei:
                                boxes_it = boxes_it + split_box[box]
                            else:
                                box.active = False
                                boxes_it.append(box)
                    else:
                        if self.ei_bound(box) >= self.max_ei:
                            boxes_it = boxes_it + split_box[box]
                        else:
                            box.active = False
                            boxes_it.append(box)
            self.boxes = boxes_it
        best_x, max_ei = self.sample_ei_active_boxes_centers()
        if max_ei > self.max_ei:
            self.max_ei = max_ei
            self.best_x = best_x

        return self.best_x, self.max_ei
