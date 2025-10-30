# tamuBO

TAMU Bayesian Optimization tools

## Install

Use a lightweight setup:

### Prerequisites

* Python 3.9+
* Dependencies: `scikit-learn`, `pandas`, `scipy`, `numpy`, etc.

### Setup

```bash
git clone https://github.com/juanesfco/tamubo.git
cd tamubo
python -m venv venv
source venv/bin/activate # On Windows: venv/Scripts/activate
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

## Repo layout

```bash
tamubo
├── development
│   ├── boTorch.ipynb
│   ├── exactBO.ipynb
│   ├── Figures
│   ├── gpflow.ipynb
│   ├── gpNumpy.ipynb
│   ├── gpytorch.ipynb
│   ├── pytorch.ipynb
│   ├── runpython.sh
│   └── torchNumpyExample.py
├── examples
│   ├── demo_2d.py
│   └── test.ipynb
├── pyproject.toml
├── README.md
├── requirements.txt
└── src
    └── tamubo
        ├── __init__.py
        ├── exactbo
        │   ├── __init__.py
        │   ├── ei_bounds.py
        │   ├── ei.py
        │   ├── kernel_bounds.py
        │   ├── loop_exactbo.py
        │   ├── loop_partition.py
        │   ├── partition.py
        │   └── posterior_bounds.py
        ├── gpugp
        │   ├── __init__.py
        │   ├── _gp.py
        │   ├── kernels.py
        │   ├── test.ipynb
        │   ├── testObjSpace.ipynb
        │   └── utils.py
        └── mobbo
            ├── __init__.py
            ├── acquisitionFunc.py
            ├── current_rep.csv
            ├── gpModel.py
            ├── hv_average_rand.csv
            ├── hv_curr_rand.csv
            ├── hv_stdv_rand.csv
            ├── main_BO.py
            ├── multiobjective.py
            └── test.ipynb
```

---

## ExactBO

A minimal, modular scaffold for **Exact Bayesian Optimization** with four core pieces:

1) **Domain partition** (axis-aligned hyperboxes)
2) **EI bounds** estimation over boxes  
3) **Partition loop** to find max EI **without** updating the GP  
4) **Outer exactBO loop** to drive to the global optimum

### Quickstart (2D demo)

The library expects any model with:

```python
model.fit(X, y)
mu, std = model.predict(X, return_std=True)
```

(e.g., `sklearn.gaussian_process.GaussianProcessRegressor`)

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from exactbo.partition import Box, split_box    # if you expose Box in partition.py
from exactbo.loop_partition import PartitionMaxEISearch
from exactbo.loop_exactbo import ExactBOLoop

# Bounds [0,1]^2
bounds = np.array([[0.0, 1.0],
                   [0.0, 1.0]])

# Toy objective
def f(X):
    X = np.atleast_2d(X)
    return ((X - 0.3)**2).sum(axis=1, keepdims=True) + 0.1*np.sin(12*X).sum(axis=1, keepdims=True)

# Initial data
rng = np.random.default_rng(0)
X0 = rng.random((8, 2))
y0 = f(X0)

# GP model (scikit-learn)
kernel = C(1.0) * RBF(length_scale=0.3)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

# Outer BO loop (your loop_exactbo should call gp.fit / gp.predict internally)
bo = ExactBOLoop(model=gp, bounds=bounds)
bo._evaluate_oracle = f  # simple pluggable oracle for the demo
res = bo.run(X0=X0, y0=y0, budget=20)

print("Best y:", res.y_best)
print("Best x:", res.x_best)
```

### Key APIs (minimal)

**`ei.expected_improvement(mu, sigma, f_best) -> Array`**
EI for minimization (broadcasts over n×1 arrays).

**`ei_bounds.ei_bounds_from_mu_sigma_intervals(mu_lo, mu_hi, var_lo, var_hi, f_best)`**
Conservative EI bounds via corner evaluation; tighten by refining partitions.

**`partition.Box(bounds)`**
Holds a `(d,2)` array with `[lo, hi]` per dimension; convenience `center`, `width`.

**`loop_partition.PartitionMaxEISearch(model, f_best, init_boxes)`**
Branch-and-bound over boxes:

* `run(max_iters=...) -> (x_best, ei_best)`

**`loop_exactbo.ExactBOLoop(model, bounds)`**
Outer loop:

* `run(X0, y0, budget) -> BOResult(X, y, x_best, y_best)`

### Roadmap (tiny)

* Revise bounds.
* Add termination: stop when `max(EI_upper) ≤ ε`
* (Optional) gpytorch backend & per-dim input-noise controls

---

## gpuGP

---

## MOBBO

---

## License

Distributed under the [MIT License](https://opensource.org/licenses/MIT).

## Contact

* **Juan E. Flórez-Coronel** - [juan.florez@tamu.edu](mailto:juan.florez@tamu.edu)

---
