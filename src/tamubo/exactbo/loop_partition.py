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

    def __init__(self, model, init_box: Box, grid: Array, precision: Array, log: dict | bool = False):
        self.model = model
        self.boxes: List[Box] = [init_box]
        self.grid = grid
        self.precision = precision
        self.log = log
        self.max_ei: float = 0.0
        self.best_x: Array | None = None
    
    def sample_ei_active_boxes(self, percentage: float = 0.1, seed = None) -> float:
        active_boxes_bounds = []
        sampled_boxes_bounds = []
        for box in self.boxes:
            if box.active:
                active_boxes_bounds.append(box.bounds)
            if box.sampled:
                sampled_boxes_bounds.append(box.bounds)
        mask_big = hypermask(np.array(active_boxes_bounds), self.grid)

        rng = np.random.default_rng(seed)
        grid_sample = rng.choice(self.grid[mask_big], size=int(round(percentage*mask_big.sum())), replace=False)
        ei_sample = expected_improvement(grid_sample, self.model)
        
        argsort_ei_sample = np.argsort(ei_sample)
        found = False
        i = 0
        best_x = None
        max_ei = 0
        while not found and ((i*(-1)) < len(ei_sample)):
            i -= 1
            best_id = argsort_ei_sample[i]
            best_x = grid_sample[best_id]
            mask_sampled = hypermask(np.array(sampled_boxes_bounds),best_x.reshape((1,self.grid.shape[1])))
            if mask_sampled.sum() == 0:
                found = True
                max_ei = ei_sample[best_id]
        
        #best_id = np.argmax(ei_sample)
        #best_x = grid_sample[best_id]
        #max_ei = ei_sample[best_id]
        return best_x, max_ei
    
    def sample_ei_active_boxes_centers(self) -> float:
        active_boxes_centers = []
        for box in self.boxes:
            if box.active:
                active_boxes_centers.append(box.center)
        active_boxes_centers = np.array(active_boxes_centers)
        ei_sample = expected_improvement(active_boxes_centers, self.model)

        argsort_ei_sample = np.argsort(ei_sample)
        found = False
        i = 0
        while not found and ((i*(-1)) < len(ei_sample)):
            i -= 1
            best_id = argsort_ei_sample[i]
            best_x = active_boxes_centers[best_id]
            if best_x not in self.model.X_train_:
                found = True
                max_ei = ei_sample[best_id]

        #best_id = np.argmax(ei_sample)
        #best_x = active_boxes_centers[best_id]
        #max_ei = ei_sample[best_id]
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
        if self.log:
            iteration_key = list(self.log.keys())[-1]
            self.log[iteration_key]["ploop_start"] = {"boxes": self.boxes, "best_x": self.best_x, "max_ei": self.max_ei}
        
        it = 0
        flag = True
        #prev_box_count = len(self.boxes)
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
            
            if self.log:
                self.log[iteration_key][f"ploop_{it - 1}"] = {"boxes": self.boxes, "best_x": self.best_x, "max_ei": self.max_ei}

            if len(self.boxes) == len(boxes_it):
                flag = False

            self.boxes = boxes_it
        
        best_x, max_ei = self.sample_ei_active_boxes_centers()
        if max_ei > self.max_ei:
            self.max_ei = max_ei
            self.best_x = best_x
        
        if self.log:
            self.log[iteration_key]["ploop_final"] = {"boxes": self.boxes, "best_x": self.best_x, "max_ei": self.max_ei}
            #self.log[iteration_key]["ploop_final"] = {"boxes": self.boxes, "best_x": best_x, "max_ei": max_ei}

        return self.best_x, self.max_ei
        #return best_x, max_ei
