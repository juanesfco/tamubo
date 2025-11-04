from __future__ import annotations
import numpy as np
from typing import List

from .partition import Box, Boxes, split_box, hypermask
from .ei import expected_improvement
from .bounds import ei_bounds

Array = np.ndarray

class PartitionMaxEISearch:
    """
    Branch-and-bound search for max EI over partitioned boxes WITHOUT updating the GP.

    Assumes `model.predict(X, return_std=True) -> (mu, std)` with (n,1) arrays.
    """

    def __init__(self, model, init_box: Box, grid: Array, precision: Array, log: dict | bool = False):
        self.model = model
        self.boxes: Boxes = Boxes([init_box])
        self.grid = grid
        self.precision = precision
        self.log = log
        self.max_ei: float = 0.0
        self.best_x: Array | None = None
    
    def random_sample_ei_active_non_sampled_boxes(self, percentage: float = 0.1, seed = None) -> float:
        active_boxes_mask = self.boxes.active
        non_sampled_boxes_mask = ~self.boxes.sampled
        active_non_sampled_boxes_mask = active_boxes_mask & non_sampled_boxes_mask
        
        if np.any(active_non_sampled_boxes_mask):
            mask_boxes = Boxes(self.boxes.items[active_non_sampled_boxes_mask])
        else:
            mask_boxes = Boxes(self.boxes.items[active_boxes_mask])

        mask_grid = hypermask(mask_boxes.bounds, self.grid)

        rng = np.random.default_rng(seed)
        grid_sample = rng.choice(self.grid[mask_grid], size=int(round(percentage*mask_grid.sum())), replace=False)
        ei_sample = expected_improvement(grid_sample, self.model)
        
        best_id = np.argmax(ei_sample)
        best_x = grid_sample[best_id]
        max_ei = ei_sample[best_id]

        return best_x, max_ei
    
    def sample_ei_active_non_sampled_boxes_centers(self) -> float:
        active_boxes_mask = self.boxes.active
        non_sampled_boxes_mask = ~self.boxes.sampled

        active_non_sampled_boxes = Boxes(self.boxes.items[active_boxes_mask & non_sampled_boxes_mask])

        ei_sample = expected_improvement(active_non_sampled_boxes.centers, self.model)

        best_id = np.argmax(ei_sample)
        best_x = active_non_sampled_boxes.centers[best_id]
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

    def run(self, max_iters: int = 100):
        it = 0
        flag = True
        
        while it < max_iters and flag:
            boxes_it = Boxes()
            it += 1
            best_x, max_ei = self.random_sample_ei_active_non_sampled_boxes()
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
                                boxes_it.extend(split_box(box))
                        else:
                            box.sampled = False
                            ei = ei_bounds(box, self.model)
                            box.ei = ei

                            if ei.hi < self.max_ei:
                                box.active = False
                                boxes_it.append(box)
                            else:
                                if np.all(box.width <= self.precision):    
                                    boxes_it.append(box)
                                else:
                                    boxes_it.extend(split_box(box))
                    else:
                        ei = ei_bounds(box, self.model)
                        box.ei = ei

                        if ei.hi < self.max_ei:
                            box.active = False
                            boxes_it.append(box)
                        else:
                            if np.all(box.width <= self.precision):    
                                boxes_it.append(box)
                            else:
                                boxes_it.extend(split_box(box))
            
            if self.log:
                iteration_key = list(self.log.keys())[-1]
                self.log[iteration_key][f"ploop_{it - 1}"] = {"boxes": self.boxes, "best_x": self.best_x, "max_ei": self.max_ei}

            if len(self.boxes) == len(boxes_it):
                flag = False

            self.boxes = boxes_it
        
        best_x, max_ei = self.sample_ei_active_non_sampled_boxes_centers()
        if max_ei > self.max_ei:
            self.max_ei = max_ei
            self.best_x = best_x
        
        if self.log:
            self.log[iteration_key]["ploop_final"] = {"boxes": self.boxes, "best_x": self.best_x, "max_ei": self.max_ei}
            #self.log[iteration_key]["ploop_final"] = {"boxes": self.boxes, "best_x": best_x, "max_ei": max_ei}

        return self.best_x, self.max_ei
        #return best_x, max_ei
