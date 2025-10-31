from __future__ import annotations
import numpy as np
from typing import List

from .partition import Box, split_box, hypermask
from .ei import expected_improvement
from .bounds import ei_bounds

Array = np.ndarray

class PartitionMaxEISearch:
    """
    Branch-and-bound search for max EI over partitioned boxes WITHOUT updating the GP.

    Assumes `model.predict(X, return_std=True) -> (mu, std)` with (n,1) arrays.
    """

    def __init__(self, model, init_box: Box, grid: Array, precision: Array, verbose: bool = False):
        self.model = model
        self.boxes: List[Box] = [init_box]
        self.grid = grid
        self.precision = precision
        self.verbose = verbose
        self.max_ei: float = 0.0
        self.best_x: Array | None = None
    
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
    
    def check_sampled(self, box: Box):
        X = self.model.X_train_
        bounds = np.array([box.bounds]) # To ensure shape (1,d,2)
        mask = hypermask(bounds, X)
        if np.sum(mask) > 0:
            return True
        else:
            return False

    def run(self, max_iters: int = 10):
        it = 0
        flag = True
        prev_box_count = len(self.boxes)
        while it < max_iters and flag:
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
                        if self.check_sampled(box):
                            if np.all(box.width <= self.precision):
                                boxes_it.append(box)
                            else:
                                boxes_it = boxes_it + split_box(box)
                        else:
                            box.sampled = False
                            if np.all(box.width <= self.precision):
                                boxes_it.append(box)
                            else:
                                ei = ei_bounds(box, self.model)
                                box.ei = ei
                                if ei.hi >= self.max_ei:
                                    boxes_it = boxes_it + split_box(box)
                                else:
                                    box.active = False
                                    boxes_it.append(box)
                    else:
                        if np.all(box.width <= self.precision):
                            boxes_it.append(box)
                        else:
                            ei = ei_bounds(box, self.model)
                            box.ei = ei
                            if ei.hi >= self.max_ei:
                                boxes_it = boxes_it + split_box(box)
                            else:
                                box.active = False
                                boxes_it.append(box)
            self.boxes = boxes_it

            if prev_box_count == len(self.boxes):
                flag = False
            else:
                prev_box_count = len(self.boxes)
            
            if self.verbose:
                print("Partition iteration: ", it-1)
        
        best_x, max_ei = self.sample_ei_active_boxes_centers()
        if max_ei > self.max_ei:
            self.max_ei = max_ei
            self.best_x = best_x

        return self.best_x, self.max_ei
