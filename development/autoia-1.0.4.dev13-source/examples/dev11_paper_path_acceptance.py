"""Standalone acceptance test for AutoIA 1.0.4.dev11 paper-comparison path."""

import time
import numpy as np
from scipy.optimize import fsolve, linprog

import autoia
from autoia import Interval, autoia as run_autoia, prepare


def response(x):
    def residual(y):
        return np.array([y[0] - x], dtype=object)
    return fsolve(residual, [0.5])[0]


class Holder:
    pass


def make_model():
    holder = Holder()
    holder.g = response

    def model(p):
        x = p[0]
        gx = holder.g(x)  # direct function-valued attribute access
        return linprog(
            c=[-1.0],
            A_ub=[[1.0]],
            b_ub=[gx],
            bounds=[(0.0, 1.0)],
        ).fun

    return model


def boxes():
    return [
        [Interval(0.20, 0.30)],
        [Interval(0.30, 0.40)],
        [Interval(0.40, 0.50)],
        [Interval(0.50, 0.60)],
        [Interval(0.60, 0.70)],
        [Interval(0.70, 0.80)],
    ]


def main():
    print("AutoIA version:", autoia.__version__)
    assert autoia.__version__ == "1.0.4.dev11"
    model = make_model()

    print("\nTEST 1 -- direct obj.g(x) dispatch")
    r = run_autoia(model, [Interval(0.2, 0.6)])
    print(r)
    assert r.machine_rigorous
    assert r.lower[0] <= -0.6 and r.upper[0] >= -0.2
    assert any(c.get("operation") == "scipy.optimize.fsolve" for c in r.certificates)
    assert any(c.get("operation") == "scipy.optimize.linprog" for c in r.certificates)
    print("PASS")

    print("\nTEST 2 -- prepare once + persistent process batches")
    B = boxes()
    serial = prepare(model)
    ref_lo, ref_hi = serial.enclose_many_bounds(
        B, workers=1, parallel="serial", use_cache=False
    )

    prepared = prepare(model)
    t0 = time.perf_counter()
    with prepared.manybox_executor(
        workers=2, parallel="process", chunk_size=1, use_cache=False
    ) as pool:
        lo1, hi1, info1 = pool.enclose_many_bounds(B[:3], return_info=True)
        lo2, hi2, info2 = pool.enclose_many_bounds(B[3:], return_info=True)
    elapsed = time.perf_counter() - t0

    got_lo = np.vstack([lo1, lo2])
    got_hi = np.vstack([hi1, hi2])
    assert np.array_equal(got_lo, ref_lo)
    assert np.array_equal(got_hi, ref_hi)
    assert info1.resolved_parallel == "process"
    assert info2.resolved_parallel == "process"
    assert info1.machine_rigorous and info2.machine_rigorous
    assert info2.scheduling_note == "persistent-manybox-executor"
    print("process results are bit-for-bit identical to prepared serial reference")
    print("second-batch execution info:", info2)
    print(f"two-batch wall time including first spawn: {elapsed:.6f} s")
    print("PASS")

    print("\nALL DEV11 PAPER-PATH ACCEPTANCE TESTS PASSED")


if __name__ == "__main__":
    main()
