"""08 — Normal PDF/CDF plus math.log10 and np.log10 support."""
import math
import numpy as np
from autoia import Interval, autoia, standard_normal_pdf, standard_normal_cdf


def model(p):
    x = p[0]
    return np.array([
        standard_normal_pdf(x),
        standard_normal_cdf(x),
        math.log10(2.0 + x),
        np.log10(3.0 + x),
    ], dtype=object)


if __name__ == "__main__":
    result = autoia(model, [Interval(-0.5, 0.5)])
    print(result)
