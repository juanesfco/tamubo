"""09 — Typed fail-closed behavior."""
import numpy as np
from autoia import AutoIAError, Interval, autoia


def unsafe_model(p):
    x = p[0]
    return 1.0 / x


def sqrt_model(p):
    x = p[0]
    return np.sqrt(x)


if __name__ == "__main__":
    try:
        autoia(unsafe_model, [Interval(-1.0, 1.0)])
    except AutoIAError as exc:
        print("Expected failure:", type(exc).__name__)
        print(exc)
        if getattr(exc, "context", None):
            print("context:", exc.context)

    # The exact physical boundary x=0 is allowed for sqrt. AutoIA proves the
    # physical-domain condition rather than clipping a structural enclosure.
    result = autoia(sqrt_model, [Interval(0.0, 4.0)])
    print("sqrt([0,4]) certified:", result.lower, result.upper)
