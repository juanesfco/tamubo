"""Automatic configuration and expert override example.

Run from the project root with:
    python examples/automatic_configuration.py
"""
import numpy as np
from autoia import autoia, Interval, ConfigurationPolicy


def model(p):
    return np.exp(3.0*p[0])


box = [Interval(-1.0, 1.0)]

# Version-1 novice behavior: automatic configuration is the zero-knob default.
automatic = autoia(model, box)
print("Automatic novice call:")
print(automatic)
print("\nWhy AutoIA made those configuration choices:")
print(automatic.explain_configuration())

# Expert/reproducibility behavior: explicitly supplying order or representation
# requests one fixed configuration unless configuration_policy is also explicit.
fixed = autoia(model, box, order=4)
print("\nExplicit fixed order-4 run:")
print(fixed)

# Research oracle: deliberately run the bounded reference portfolio.
reference = autoia(
    model,
    box,
    configuration_policy=ConfigurationPolicy(mode="reference"),
)
print("\nExpensive reference portfolio:")
print(reference)
