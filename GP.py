"""Gaussian processes regression."""

# Author: Juan E Florez-Coronel



class GPR:
    """"Gaussian process regression (GPR).

    The implementation is based on Algorithm 2.1 of [RW2006]_.

    Uses PyTorch Tensors when to take advantage of GPU computing
    capabilities when possible.

    Parameters
    ----------
    kernel : kernel instance, default=None
        The kernel specifying the covariance function of the GP. If None is
        passed, the squared-exponential covariance function is used as 
        default.

    sigma_n_squared : float or ndarray of shape (n_samples,), default=1e-10
        Value added to the diagonal of the kernel matrix during fitting.
        This can prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
        It can also be interpreted as the variance of additional Gaussian
        measurement noise on the training observations. If an array is passed, 
        it must have the same number of entries as the data used for fitting 
        and is used as datapoint-dependent noise level.

    Attributes
    ----------
    X_train_ : array-like of shape (n_samples, n_features) or list of object
        Feature vectors or other representations of training data (also
        required for prediction).

    y_train_ : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values in training data (also required for prediction).

    kernel_ : kernel instance
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters.

    References
    ----------
    .. [RW2006] `Carl E. Rasmussen and Christopher K.I. Williams,
       "Gaussian Processes for Machine Learning",
       MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_

    Examples
    --------
    >>> do later
    """

    def __init__(self, kernel=None, alpha=1e-10):
        self.kernel = kernel
        self.alpha = alpha

    def fit(self, X, y):
        """Fit Gaussian process regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            GaussianProcessRegressor class instance.
        """
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = # Add Kernel
        else:
            self.kernel_ = clone(self.kernel)

        