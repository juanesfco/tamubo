"""01 — Smooth scalar model using ordinary NumPy elementary functions."""
import numpy as np
from autoia import Interval, autoia


def model(p):
    x = p[0]
    return np.exp(x) + np.sin(x) + x * x


if __name__ == "__main__":
    box = [Interval(-0.2, 0.3)]
    result = autoia(model, box)
    print(result)

    # Sampling is diagnostic only.  It is not part of the proof.
    xs = np.linspace(-0.2, 0.3, 1001)
    vals = np.exp(xs) + np.sin(xs) + xs * xs
    print("sample min/max (diagnostic only):", vals.min(), vals.max())
