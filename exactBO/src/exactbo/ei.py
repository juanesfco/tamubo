import numpy as np
from scipy.stats import norm

Array = np.ndarray

def expected_improvement(mu: Array, sigma: Array, f_best: float) -> Array:
    """
    EI for minimization. mu,sigma are (n,1) or (n,) arrays.
    """
    mu = np.atleast_2d(mu).reshape(-1, 1)
    sigma = np.maximum(np.atleast_2d(sigma).reshape(-1, 1), 1e-12)
    z = (f_best - mu) / sigma
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
