"""Additional common NumPy elementary functions supported transparently by AutoIA."""
import numpy as np
from autoia import autoia, Interval


def engineering_model(p):
    x = p[0]
    return np.array([
        np.tanh(x),
        np.sinh(x),
        np.arctan(x),
        np.abs(x),
        np.log1p(x),
        np.cosh(x),
    ], dtype=object)


if __name__ == "__main__":
    box = [Interval.enclosing("-0.4", "0.7")]
    result = autoia(engineering_model, box)
    print(result)
    print("outputs: tanh, sinh, arctan, abs, log1p, cosh")
