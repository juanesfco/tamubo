"""Example 42 -- persistent prepared execution for a B&B-style node queue.

This example exercises the paper-comparison execution pattern added in
AutoIA 1.0.4.dev11:

* a response function stored directly on a captured object attribute;
* nested scipy.optimize.fsolve inside that response;
* scipy.optimize.linprog inside the node model;
* prepare(model) once;
* several successive batches sent through one persistent process pool.

The process pool changes scheduling only.  Every node is independently certified
by the normal CPU-binary64 AutoIA proof path.
"""

import time
import numpy as np
from scipy.optimize import fsolve, linprog

from autoia import Interval, prepare


def response(x):
    # Deliberately use an ordinary SciPy implicit solve.
    def residual(y):
        return np.array([y[0] - x], dtype=object)
    return fsolve(residual, [0.5])[0]


class Problem:
    pass


def make_node_model():
    problem = Problem()
    # Important dev11 dispatch case: a function stored on the INSTANCE, not a
    # class-defined method and not hoisted into a separate closure variable.
    problem.g = response

    def node_model(p):
        x = p[0]
        gx = problem.g(x)
        # min -z s.t. z <= g(x), 0 <= z <= 1.  Here g(x)=x, so V(x)=-x.
        return linprog(
            c=[-1.0],
            A_ub=[[1.0]],
            b_ub=[gx],
            bounds=[(0.0, 1.0)],
        ).fun

    return node_model


def make_boxes():
    edges = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    return [
        [Interval(edges[i], edges[i + 1])]
        for i in range(len(edges) - 1)
    ]


def main():
    model = make_node_model()
    boxes = make_boxes()
    batch1 = boxes[:3]
    batch2 = boxes[3:]

    prepared = prepare(model)

    t0 = time.perf_counter()
    with prepared.manybox_executor(
        workers=2,
        parallel="process",
        chunk_size=1,
        use_cache=False,
    ) as pool:
        lo1, hi1, info1 = pool.enclose_many_bounds(batch1, return_info=True)
        lo2, hi2, info2 = pool.enclose_many_bounds(batch2, return_info=True)
    elapsed = time.perf_counter() - t0

    lower = np.vstack([lo1, lo2])[:, 0]
    upper = np.vstack([hi1, hi2])[:, 0]

    print("AutoIA persistent B&B-style execution")
    print("resolved parallel mode:", info2.resolved_parallel)
    print("resolved workers:", info2.resolved_workers)
    print("successive calls:", 2)
    print("total boxes:", len(boxes))
    print(f"wall time including first process startup: {elapsed:.6f} s")
    print()
    for i, (box, lo, hi) in enumerate(zip(boxes, lower, upper), start=1):
        # Exact V([a,b])=[-b,-a].
        a, b = box[0].lo, box[0].hi
        print(
            f"node {i}: x=[{a:.2f},{b:.2f}]  "
            f"certified=[{lo:.12g},{hi:.12g}]  exact=[{-b:.2f},{-a:.2f}]"
        )
        assert lo <= -b <= hi
        assert lo <= -a <= hi

    assert info1.machine_rigorous and info2.machine_rigorous
    print("\nPASS: all nodes independently machine-rigorous.")


if __name__ == "__main__":
    main()
