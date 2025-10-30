"""
Minimal 2D example using scikit-learn GP.
Run:
    python -m examples.demo_2d
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from tamubo.exactbo import ExactBOLoop

"""
# Domain: [0,1]^2
bounds = np.array([[0.0, 1.0],
                   [0.0, 1.0]])

# Toy objective (one global min + wiggles)
def f(X):
    X = np.atleast_2d(X)
    return ((X - 0.35) ** 2).sum(axis=1, keepdims=True) + 0.05 * np.sin(12 * X).sum(axis=1, keepdims=True)

# Initial data
rng = np.random.default_rng(0)
X0 = rng.random((8, 2))
y0 = f(X0)

# GP model
kernel = C(1.0) * RBF(length_scale=0.25)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

# BO loop
bo = ExactBOLoop(model=gp, bounds=bounds)
bo.set_oracle(f)
res = bo.run(X0=X0, y0=y0, budget=20)

print("Best y:", res.y_best)
print("Best x:", res.x_best.ravel())
"""