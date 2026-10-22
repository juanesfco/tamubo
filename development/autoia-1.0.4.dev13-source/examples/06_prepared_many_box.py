"""06 — Prepared repeated-query and many-box execution."""
import numpy as np
from autoia import Interval, prepare


def model(p):
    x, y = p
    return np.array([np.exp(x) + y * y, x * y], dtype=object)


if __name__ == "__main__":
    # prepare(model) uses the same automatic novice configuration semantics as
    # autoia(model, box), while reusing preparation and cached certificates.
    session = prepare(model, cache_max_entries=32)

    parent = [Interval(-1.0, 1.0), Interval(0.0, 2.0)]
    child = [Interval(-0.25, 0.25), Interval(0.5, 1.0)]
    r_parent = session.certify(parent)
    r_child = session.certify(child)
    r_repeat = session.certify(child)

    print("parent cache info:", r_parent.cache_info)
    print("child cache info:", r_child.cache_info)
    print("repeat cache info:", r_repeat.cache_info)
    print("session stats:", session.stats)

    boxes = [
        [Interval(-0.5, -0.4), Interval(0.9, 1.0)],
        [Interval(0.0, 0.1), Interval(1.0, 1.1)],
        [Interval(0.4, 0.5), Interval(1.1, 1.2)],
    ]
    lo, hi = session.certify_many_bounds(boxes, parallel="serial")
    print("many-box lower:\n", lo)
    print("many-box upper:\n", hi)
