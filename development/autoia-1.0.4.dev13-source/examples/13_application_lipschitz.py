"""13 — a user-supplied Lipschitz theorem."""
import math
import numpy as np
from autoia.application import ApplicationBox, application_theorem


def f_point(x0, x1, cert):
    # f(x0,x1)=sin(x0)+x1^2 evaluated rigorously at point constants.
    X0 = cert.point(x0)
    X1 = cert.point(x1)
    return cert.sin(X0) + cert.square(X1)


@application_theorem(name="known Lipschitz constant", target="f")
def lipschitz_upper(box, cert, L):
    # User theorem: |f(x)-f(c)| <= L ||x-c||_2 on this application domain.
    cert.require(L >= 0.0, "Lipschitz constant L is nonnegative")
    c = 0.5 * box.lower + 0.5 * box.upper

    # r is the farthest Euclidean distance from the box center.  Compute it
    # through certified arithmetic rather than ordinary floating arithmetic.
    halfwidths = [
        cert.maximum(cert.point(cj) - lj, cert.point(uj) - cj)
        for cj, lj, uj in zip(c, box.lower, box.upper)
    ]
    r2 = cert.sum(cert.square(h) for h in halfwidths)
    r = cert.sqrt(r2)
    fc = f_point(c[0], c[1], cert)
    return fc + cert.point(L) * r


if __name__ == "__main__":
    box = ApplicationBox(np.array([0.0, -0.2]), np.array([0.4, 0.3]))
    # For this toy domain L=2 is a deliberately simple safe constant.
    bound = lipschitz_upper(box, L=2.0)
    print(bound)
