"""07 — Shared auxiliary generators and automatic cancellation recovery."""
import numpy as np
from autoia import Interval, autoia


def model(p):
    x = p[0]
    y = x * x
    return np.array([y, y - y], dtype=object)


if __name__ == "__main__":
    box = [Interval(-1.0, 1.0)]

    # Normal user call.  The automatic controller detects reused approximation
    # error after physical dependence cancels and may retry with generators.
    automatic = autoia(model, box)
    print("automatic result:")
    print(automatic)
    print(automatic.explain_configuration())
    print("automatic width of y-y:", automatic.width[1])

    # Expert diagnostic comparison.  Order 1 deliberately forces x^2 outside
    # the retained physical polynomial so the difference is visually obvious.
    polynomial = autoia(model, box, order=1, representation="polynomial")
    generator = autoia(model, box, order=1, representation="generator")

    print("\nfixed order-1 polynomial representation:")
    print(polynomial)
    print("fixed order-1 generator representation:")
    print(generator)
    print("generator ids:", generator.generator_ids)
    print("generator matrix:\n", generator.generator_matrix)
    print("width of y-y, polynomial:", polynomial.width[1])
    print("width of y-y, generator:", generator.width[1])
