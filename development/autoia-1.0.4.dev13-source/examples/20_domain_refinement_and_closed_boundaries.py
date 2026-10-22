"""20 — Closed-boundary comparisons and automatic proof-driven refinement.

This example demonstrates two V1 hardening behaviors found by adversarial
heat-exchanger testing:
1. strict comparisons respect exact closed-box endpoints; and
2. a broad reciprocal proof may trigger physical-box subdivision instead of an
   immediate false domain failure.
"""
import numpy as np
from autoia import Interval, autoia


def closed_boundary_model(p):
    c = p[0]
    if c > 1.0:
        raise ValueError("This branch is physically impossible on c <= 1.")
    if c < 1.0:
        return c*c
    return 1.0


def shell_and_tube_effectiveness(p):
    NTU, Cr = p
    shells = 2.0
    NTU1 = NTU / shells
    root = np.sqrt(1.0 + Cr*Cr)
    e = np.exp(-NTU1*root)
    effectiveness1 = 2.0 / (1.0 + Cr + root*(1.0 + e)/(1.0 - e))
    term = (1.0 - effectiveness1*Cr) / (1.0 - effectiveness1)
    return (term**shells - 1.0) / (term**shells - Cr)


if __name__ == "__main__":
    endpoint = autoia(closed_boundary_model, [Interval(0.95, 1.0)])
    print("closed endpoint result:")
    print(endpoint)
    print("feasible paths:", len(getattr(endpoint, "branch_results", ())))

    # This broad box previously stopped at an intermediate reciprocal whose
    # enclosure crossed zero.  Automatic mode now refines the physical box and
    # retries the proof before failing.
    hx = autoia(
        shell_and_tube_effectiveness,
        [Interval(1.0, 3.0), Interval(0.3, 0.8)],
    )
    print("\nshell-and-tube effectiveness broad-box certificate:")
    print(hx)
    print("machine rigorous:", hx.machine_rigorous)
    print("automatic configuration:", hx.automatic_configuration)
