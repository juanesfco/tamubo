"""00 — Smallest useful AutoIA example.

Run from the project root:
    python examples/00_smoke_test.py

The ordinary user supplies only a model and physical input intervals.  AutoIA
chooses its release configuration automatically.
"""
from autoia import Interval, autoia, __version__


def model(p):
    x = p[0]
    return x * x + 2.0 * x + 1.0


if __name__ == "__main__":
    result = autoia(model, [Interval(-0.2, 0.3)])
    print("AutoIA version:", __version__)
    print(result)
    print("lower:", result.lower)
    print("upper:", result.upper)
    print("machine_rigorous:", result.machine_rigorous)
    print("automatic_configuration:", result.automatic_configuration)
