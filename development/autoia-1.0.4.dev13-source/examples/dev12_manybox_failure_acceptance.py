"""AutoIA 1.0.4.dev12 acceptance: mixed-success B&B batches.

The middle box is deliberately infeasible.  In dev11 a single CertificationError
aborted the complete batch.  dev12 report mode preserves the two successful
sibling bounds and marks only the failed box for refinement/handling.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

import autoia
from autoia import Interval, prepare


def node_model(p):
    x = p[0]
    # min z, subject to z >= x and 0 <= z <= 1.
    # x=.3 and .7 are feasible; x=5 is infeasible.
    return linprog(
        c=[1.0],
        A_ub=[[-1.0]],
        b_ub=[-x],
        bounds=[(0.0, 1.0)],
    ).fun


def main():
    print("AutoIA version:", autoia.__version__)
    assert autoia.__version__ == "1.0.4.dev12"

    boxes = [
        [Interval(0.3, 0.3)],
        [Interval(5.0, 5.0)],
        [Interval(0.7, 0.7)],
    ]

    prepared = prepare(node_model, order=4, cache_max_entries=0)

    print("\nTEST 1 -- compact serial report mode")
    lower, upper, ok = prepared.enclose_many_bounds(
        boxes, workers=1, parallel="serial", use_cache=False,
        on_error="report",
    )
    print("lower:", lower[:, 0])
    print("upper:", upper[:, 0])
    print("ok:   ", ok)
    assert ok.tolist() == [True, False, True]
    assert lower[0, 0] <= 0.3 <= upper[0, 0]
    assert np.isnan(lower[1, 0]) and np.isnan(upper[1, 0])
    assert lower[2, 0] <= 0.7 <= upper[2, 0]
    print("PASS")

    print("\nTEST 2 -- detailed per-box failure")
    items = prepared.certify_many(boxes, use_cache=False, on_error="report")
    for item in items:
        if item.ok:
            print(f"box {item.index}: OK  {item.result.lower} .. {item.result.upper}")
        else:
            print(
                f"box {item.index}: FAIL "
                f"{item.failure.error_type} [{item.failure.error_code}] "
                f"{item.failure.message}"
            )
    assert items[1].failure.error_code == "certification-failed"
    print("PASS")

    print("\nTEST 3 -- persistent process batch with one failed sibling")
    with prepared.manybox_executor(
        workers=2,
        parallel="process",
        chunk_size=1,
        use_cache=False,
        on_error="report",
    ) as pool:
        lower, upper, ok, info = pool.enclose_many_bounds(boxes, return_info=True)
        # Run a second batch to ensure the pool survives an ordinary failed box.
        lower2, upper2, ok2, info2 = pool.enclose_many_bounds(boxes, return_info=True)

    print("first ok: ", ok)
    print("second ok:", ok2)
    print("second-batch execution info:", info2)
    assert ok.tolist() == [True, False, True]
    assert ok2.tolist() == [True, False, True]
    assert np.array_equal(np.isnan(lower), np.isnan(lower2))
    assert np.allclose(lower[ok], lower2[ok], rtol=0.0, atol=0.0)
    assert np.allclose(upper[ok], upper2[ok], rtol=0.0, atol=0.0)
    assert info2.n_succeeded == 2 and info2.n_failed == 1
    print("PASS")

    print("\nALL DEV12 MANY-BOX FAILURE-REPORTING ACCEPTANCE TESTS PASSED")


if __name__ == "__main__":
    main()
