"""Example 1: a one-line application theorem from monotonicity."""
import numpy as np
from autoia.application import ApplicationBox, application_theorem


@application_theorem(name="exp is increasing", target="f")
def exp_upper(box, cert):
    # Application theorem: exp(x) is increasing, so max_{x in [l,u]} exp(x)=exp(u).
    x_upper = cert.point(box.upper[0])
    return cert.exp(x_upper)


if __name__ == "__main__":
    box = ApplicationBox(np.array([-0.7]), np.array([1.2]))
    bound = exp_upper(box)
    print(bound)
    print("rigorous upper bound =", bound.value)
