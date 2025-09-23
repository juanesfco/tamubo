"""A set of kernels that can be used in Gaussian processes."""

# Author: Juan E Florez-Coronel

import torch
import utils

class rbf:
    """Radial basis function kernel (aka squared-exponential kernel).

    The RBF kernel is a stationary kernel. It is also known as the
    "squared exponential" kernel. It is parameterized by a length scale
    parameter :math:`l>0`, which can either be a scalar (isotropic variant
    of the kernel) or a vector with the same number of dimensions as the inputs
    X (anisotropic variant of the kernel). The kernel is given by:

    .. math::
        k(x_i, x_j) = \\exp\\left(- \\frac{d(x_i, x_j)^2}{2l^2} \\right)

    where :math:`l` is the length scale of the kernel and
    :math:`d(\\cdot,\\cdot)` is the Euclidean distance.
    For advice on how to set the length scale parameter, see e.g. [1]_.

    This kernel is infinitely differentiable, which implies that GPs with this
    kernel as covariance function have mean square derivatives of all orders,
    and are thus very smooth.
    See [2]_, Chapter 4, Section 4.2, for further details of the RBF kernel.

    Uses PyTorch Tensors when to take advantage of GPU computing
    capabilities when possible.

    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of length_scale defines the length-scale of the respective feature 
        dimension.

    sigma_f_squared : float or ndarray of shape (n_features,), default=1.0
        The signal variance of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of sigma_f_squared defines the signal variance of the respective feature 
        dimension.

    References
    ----------
    .. [1] `David Duvenaud (2014). "The Kernel Cookbook:
        Advice on Covariance functions".
        <https://www.cs.toronto.edu/~duvenaud/cookbook/>`_

    .. [2] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <http://www.gaussianprocess.org/gpml/>`_

    Examples
    --------
    >>> do later
    """
    def __init__(self, length_scale=1.0, sigma_f_squared=1.0):
        self.length_scale = length_scale
        self.sigma_f_squared = sigma_f_squared

    def __call__(self, X, Y=None, grad=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, 
            k(X, X) if evaluated instead.

        grad : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed. Only when Y is None.

        Returns
        -------
        K : Tensor of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : Tensor of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `grad`
            is True.
        """

        # Later: Change to parallel matrix operations:
        # https://discuss.pytorch.org/t/matmul-on-multiple-gpus/33122/2

        X = utils.numpyToTorch(X)

        if Y is None:
            dists = torch.pdist(X/self.length_scale)**2
            K = self.sigma_f_squared*torch.exp(-0.5*dists)
            K = utils.torch_squareform(K)
            K.fill_diagonal_(1)

        else:
            if grad:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            Y = utils.numpyToTorch(Y)

            dists = torch.cdist(X/self.length_scale,Y/self.length_scale)**2
            K = self.sigma_f_squared*torch.exp(-0.5*dists)

        if grad:
            K_grad = (K*utils.torch_squareform(dists))[:, :, None]
            return K, K_grad
        else:
            return K

        