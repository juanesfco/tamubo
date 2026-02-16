import numpy as np
from scipy.stats import norm

Array = np.ndarray

def expected_improvement(X: Array, model) -> Array:
    """
    EI for minimization. X is input, model should have .predict method
    and ._y_train attributes.
    """
    # Extract y from model
    y = np.asarray(model.y_train_, dtype=float).ravel()
    # To undo normalization
    y_train_std = np.asarray(model._y_train_std).item()
    y_train_mean = np.asarray(model._y_train_mean).item()
    y_min = y_train_std*np.min(y) + y_train_mean
    # Extract mu, sigma from model
    mu, sigma = model.predict(X, return_std=True)
    sigma = sigma.reshape(-1,1)
    mu = mu.reshape(-1,1)
    with np.errstate(divide='warn'):
        Z = (y_min - mu) / sigma # to minimize
        ei = (y_min - mu) * norm.cdf(Z) + sigma * norm.pdf(Z) # to minimize
        ei[sigma==0.0] = 0.0
    return ei.ravel()
