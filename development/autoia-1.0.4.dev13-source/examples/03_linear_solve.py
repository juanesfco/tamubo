"""03 — Parameter-dependent semantic linear solve using np.linalg.solve."""
import numpy as np
from autoia import Interval, autoia


def spring_model(p):
    k1, k2, force = p
    A = np.array([
        [k1 + k2, -k2],
        [-k2, k2],
    ])
    b = np.array([force, 0.0])
    return np.linalg.solve(A, b)


if __name__ == "__main__":
    result = autoia(
        spring_model,
        [Interval(90.0, 110.0), Interval(900.0, 1100.0), Interval(9.5, 10.5)],
    )
    print(result)
    print("number of semantic certificates:", len(result.certificates))
    if result.certificates:
        print("last certificate keys:", sorted(result.certificates[-1].keys()))
