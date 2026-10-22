"""Example 2: let AutoIA enclose intermediates, then apply an application theorem."""
import numpy as np
from autoia.application import ApplicationBox, application_theorem


def a_model(p):
    return p[0] + p[1] * p[1]


def b_model(p):
    return np.exp(-p[0] * p[1])


@application_theorem(name="monotone outer composition", target="score")
def score_upper(box, cert):
    # Suppose the application expert has proved that G(a,b)=exp(a)+sqrt(b)
    # is nondecreasing in both arguments for b>=0.
    a = cert.enclose(a_model, box)
    b = cert.enclose(b_model, box)
    cert.require(b.lower >= 0.0, "b is nonnegative on the box")

    # Evaluate G at certified upper endpoints. All proof-critical arithmetic
    # remains certified, while AutoIA chooses its own internal representation.
    return cert.exp(cert.point(a.upper)) + cert.sqrt(cert.point(b.upper))


if __name__ == "__main__":
    box = ApplicationBox(np.array([0.2, 0.4]), np.array([0.6, 0.9]))
    bound = score_upper(box)
    print(bound)
    print("checked assumptions =", bound.assumptions_checked)
