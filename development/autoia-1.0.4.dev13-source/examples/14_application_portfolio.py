"""14 — combine independent application theorems safely."""
import numpy as np
from autoia.application import ApplicationBox, application_theorem, minimum_upper_bound


@application_theorem(name="simple global theorem", target="f")
def global_bound(box, cert):
    return cert.point(10.0)


@application_theorem(name="box-aware theorem", target="f")
def local_bound(box, cert):
    x = cert.input_coordinate(box, 0)
    # Application theorem in this example: f(x) <= x + 2 on the domain.
    return x + 2.0


if __name__ == "__main__":
    box = ApplicationBox(np.array([-1.0]), np.array([3.0]))
    b1 = global_bound(box)
    b2 = local_bound(box)
    best = minimum_upper_bound(b1, b2)
    print("global theorem:", b1.value)
    print("box-aware theorem:", b2.value)
    print("portfolio upper bound:", best.value)
    print("winning theorem:", best.winner.theorem)
