"""Gaussian processes regression."""

# Author: Juan E Florez-Coronel

import kernels
import torch
import utils
from copy import deepcopy

class gpr:
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

    def __init__(self, kernel=None, sigma_n_squared=1e-10):
        self.kernel = kernel
        self.sigma_n_squared = sigma_n_squared

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
            gpr class instance.
        """
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = kernels.rbf()
        else:
            self.kernel_ = deepcopy(self.kernel)
            
        self.X_train_ = utils.numpyToTorch(X)
        self.y_train_ = utils.numpyToTorch(y).view(-1,1)

        n_samples = X.shape[0]

        # Alg. 2.1, page 19, line 1 (I guess)
        K = self.kernel_(self.X_train_)
        # Alg. 2.1, page 19, line 2
        diag_indices = torch.arange(n_samples)
        K[diag_indices, diag_indices] += self.sigma_n_squared
        self.L_ = torch.linalg.cholesky(K)
        # Alg. 2.1, page 19, line 3
        self.alpha_ = torch.cholesky_solve(self.y_train_,self.L_) #equivalent to line below with less numerical error
        #self.alpha_ = torch.linalg.solve_triangular(torch.transpose(self.L_,0,1), torch.linalg.solve_triangular(self.L_, self.y_train_, upper=False), upper=False)
        return self
    
    def predict(self, X):
        """Predict using the Gaussian process regression model.

        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, the covariance
        is returned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated.

        Returns
        -------
        y_mean : Tensor of shape (n_samples,) or (n_samples, n_targets)
            Mean of predictive distribution at query points.

        y_cov : Tensor of shape (n_samples, n_samples) or \
                (n_samples, n_samples, n_targets), optional
            Covariance of joint predictive distribution at query points.
        """

        X = utils.numpyToTorch(X)
        
        # Alg. 2.1, page 19, line 4
        K_star = self.kernel_(self.X_train_, X)
        y_mean = torch.mm(torch.transpose(K_star,0,1),self.alpha_)
        # Alg. 2.1, page 19, line 5
        v = torch.linalg.solve_triangular(self.L_, K_star, upper=False)
        # Alg. 2.1, page 19, line 6
        y_var = self.kernel_(X) - torch.mm(torch.transpose(v,0,1),v)

        return y_mean, y_var
    
    def get_hyper_params(self):
        """
        Get parameters for this GP.

        Returns
        -------
        curr_params : dict
            Parameter names mapped to their values: 
            length_scale, sigma_f_squared, sigma_n_squared
        """ 
        curr_params = {"length_scale":self.kernel_.length_scale, "sigma_f_squared":self.kernel_.sigma_f_squared, "sigma_n_squared":self.sigma_n_squared}
        return curr_params

    def set_hyper_params(self, params):
        """
        Set parameters for this GP and retrains automatically.

        Parameters
        ----------
        params : dict
            Parameter names mapped to their values: 
            length_scale, sigma_f_squared, sigma_n_squared

        Returns
        -------
        self : object
            gpr class instance.
        """ 
        self.kernel.length_scale = params['length_scale']
        self.kernel.sigma_f_squared = params['sigma_f_squared']
        self.sigma_n_squared = params['sigma_n_squared']
        self.fit(self.X_train_,self.y_train_)
        return self
    
    def log_marginal_likelihood(self, theta=None, grad=False):
        """Return log-marginal likelihood of theta for training data.

        Parameters
        ----------
        theta : Tensor of shape (1,n_kernel_params) default=None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.get_hyper_params`` is returned.
        grad : bool, default=False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.
        log_likelihood_gradient : Tensor of shape (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        if theta is None:
            if grad:
                raise ValueError("Gradient can only be evaluated for theta!=None")
            log_likelihood = -0.5*self.y_train_.T @ self.alpha_ - torch.sum(torch.log(self.L_.diag())) - 0.5*self.L_.shape[0]*torch.log(torch.tensor(2*torch.pi).double())
            return log_likelihood
        
        if grad:
            K, K_gradient = self.kernel_(self.X_train_, grad=True)
        else:
            K = self.kernel_(self.X_train_)

        n_samples = self.X_train_.shape[0]

        # Alg. 2.1, page 19, line 2
        diag_indices = torch.arange(n_samples)
        K[diag_indices, diag_indices] += self.sigma_n_squared

        L = torch.linalg.cholesky(K)
        y_train = self.y_train_

        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        alpha = torch.cholesky_solve(y_train,L)

        # Alg 2.1, page 19, line 7
        log_likelihood = -0.5*y_train.T @ alpha - torch.sum(torch.log(L.diag())) - 0.5*L.shape[0]*torch.log(torch.tensor(2*torch.pi).double())

        if grad:
            # Eq. 5.9, p. 114, and footnote 5 in p. 114
            print('hola')