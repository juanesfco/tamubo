"""Subprocess smoke for PreparedAutoIA process mode.

Kept outside the unittest runner's own __main__ so multiprocessing spawn does not
recursively execute test discovery.
"""
from __future__ import annotations
import numpy as np
from autoia import prepare, Interval


def model(p):
    x, y = p
    return np.exp(0.1*x) + x*y + 0.25*y*y


def main():
    session = prepare(model, order=3, cache_max_entries=0)
    boxes = [
        [Interval(-0.2, -0.1), Interval(-0.1, 0.1)],
        [Interval(0.0, 0.1), Interval(-0.1, 0.1)],
    ]
    serial = session.certify_many(boxes, use_cache=False)
    parallel = session.certify_many(
        boxes, workers=2, parallel="process", chunk_size=1, use_cache=False
    )
    for a, b in zip(serial, parallel):
        if not (np.array_equal(a.lower, b.lower) and np.array_equal(a.upper, b.upper)):
            raise SystemExit("process results differ")
    print("process-manybox-smoke: OK")


if __name__ == "__main__":
    main()
