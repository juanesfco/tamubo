"""Example 29: ordinary scikit-learn GaussianProcessRegressor inside AutoIA.

Requires optional package scikit-learn.  Model fitting is ordinary floating-
point scikit-learn work and is not certified.  AutoIA rigorously encloses the
supported fitted exact-GP posterior conditional on the stored fitted state.
No manual GP freeze adapter is used.
"""
import math
import numpy as np

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF
except ImportError:
    print("SKIP: example 29 requires optional package scikit-learn.")
    raise SystemExit(0)

from autoia import autoia, Interval


def engineering_model(p):
    # Ordinary training data and ordinary sklearn fitting.
    train_x = np.linspace(0.0, 1.0, 12).reshape(-1, 1)
    train_y = np.sin(2.0 * math.pi * train_x[:, 0])

    gp = GaussianProcessRegressor(
        kernel=(
            ConstantKernel(1.0, (1.0e-3, 1.0e3))
            * RBF(0.2, (1.0e-3, 1.0e3))
        ),
        alpha=1.0e-4,
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=0,
    )
    gp.fit(train_x, train_y)

    # Ordinary sklearn prediction code.  AutoIA recognizes the uncertain query
    # and substitutes the shared validated exact-GP posterior semantics.
    x = np.array([[p[0]]])
    mean, std = gp.predict(x, return_std=True)
    return mean[0], std[0]


result = autoia(
    engineering_model,
    [Interval.enclosing("0.20", "0.30")],
)
print(result)
assert result.machine_rigorous
