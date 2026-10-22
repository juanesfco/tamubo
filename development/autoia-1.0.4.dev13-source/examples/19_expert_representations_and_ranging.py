"""19 — Expert/reproducibility controls for representations and ranging.

Most users should call autoia(model, box) with no representation/order choices.
This example shows the lower-level controls used for diagnostics, reproducible
studies, and method development.
"""
import numpy as np
from autoia import Interval, autoia


def model(p):
    x, y = p
    z = x*x + 0.5*x*y + 0.1*y*y
    return np.exp(0.2*x) + z


if __name__ == "__main__":
    box = [Interval(-0.8, 0.7), Interval(-0.5, 0.9)]

    automatic = autoia(model, box)
    print("automatic zero-knob result:")
    print(automatic)
    print(automatic.explain_configuration())

    cases = [
        ("polynomial termwise", dict(representation="polynomial", order=4, range_method="termwise")),
        ("polynomial monotonicity", dict(representation="polynomial", order=4, range_method="monotonicity")),
        ("polynomial Bernstein", dict(representation="polynomial", order=4, range_method="bernstein")),
        ("interval", dict(representation="interval", order=4)),
        ("generator", dict(representation="generator", order=4)),
    ]
    for label, kwargs in cases:
        result = autoia(model, box, **kwargs)
        print(f"\n{label}")
        print("  lower:", result.lower)
        print("  upper:", result.upper)
        print("  width:", result.width)
        print("  machine rigorous:", result.machine_rigorous)
