"""02 — Vector output and common NumPy array/reduction semantics."""
import numpy as np
from autoia import Interval, autoia

COEFF = np.array([1.5, -2.0, 0.25])


def model(p):
    x, y = p
    v = np.array([x + y, x * y, x * x + 0.5 * y], dtype=object)
    weighted = np.dot(COEFF, v)
    return np.array([v[0], v[1], v[2], weighted, np.sum(v)], dtype=object)


if __name__ == "__main__":
    result = autoia(
        model,
        [Interval(0.9, 1.1), Interval(1.8, 2.2)],
    )
    print(result)
    print("shaped lower:", result.lower_shaped)
    print("shaped upper:", result.upper_shaped)
