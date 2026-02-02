# Print starting
print("Starting - Loops Script")

import cupynumeric as cp
import numpy as np
from legate.timing import time
from partition_cupynumeric import split_boxes
from bound_cupynumeric import rbf_k_bounds, mu_bounds, sigma_bounds, ei_bounds
from ei_cupynumeric import expected_improvement

def partition_loop(X_data, bounds, epsilon, gp, max_partitions):
    # X to CuPyNumeric
    X_cp = cp.array(X_data) # (N,d)
    
    # Initialize boxes
    ## One row per box (initially one box)
    ## One column per dimension
    bounds_L = cp.array([bounds[:,0]]) # (n,d)
    bounds_U = cp.array([bounds[:,1]]) # (n,d)

    # GP hyperparameters
    gp_kernel_params = gp.kernel_.get_params()
    sigma_f_2 = gp_kernel_params['k1__k1__constant_value']
    length_scale = gp_kernel_params['k1__k2__length_scale']
    alpha = cp.array(gp.alpha_)
    y_train_std = gp._y_train_std
    y_train_mean = gp._y_train_mean
    L = cp.array(gp.L_)
    y_min = cp.min(gp.y_train_)

    # Partition parameters
    N = X_data.shape[0]  # Number of data points
    d = bounds.shape[0]  # Number of dimensions
    w = bounds_U[0] - bounds_L[0]  # Bounds with per dimension (d,)
    partition = 0
    w_max = w.copy()
    ei_max = 0

    while partition < max_partitions and cp.all(w_max > epsilon):
        # Total number of boxes
        n = bounds_L.shape[0]
        
        # Bounds
        ## Kernel
        K_lo = cp.zeros((n,N))
        K_hi = cp.zeros((n,N))
        for i in range(N):
            xi = X_cp[i]
            K_lo[:,i], K_hi[:,i] = rbf_k_bounds(bounds_L.ravel(),bounds_U.ravel(),xi,n,d,sigma_f_2,length_scale,False)
        ## Mean
        mu_lo, mu_hi = mu_bounds(alpha,K_lo,K_hi,n,N,y_train_mean,y_train_std,False) # (n,) both
        ## Sigma
        sig_lo, sig_hi = sigma_bounds(K_lo,K_hi,L,n,N,sigma_f_2,y_train_std,False) # (n,) both
        ## EI
        ei_lo, ei_hi = ei_bounds(mu_lo,mu_hi,sig_lo,sig_hi,y_min,y_train_mean,y_train_std,False) # (n,) both
        
        # Compute actual EI in the center of the hyperbox with highest upper EI bound
        idx_max_ei_hi = cp.argmax(ei_hi)
        box_L = bounds_L[idx_max_ei_hi,:]  # (d,)
        box_U = bounds_U[idx_max_ei_hi,:]  # (d,)
        box_center = (box_L + box_U) / 2.0  # (d,)
        mu_pred, sigma_pred = gp.predict(cp.asnumpy(box_center).reshape(1,-1), return_std=True)
        ei_max = cp.max(ei_max,expected_improvement(mu_pred[0], sigma_pred[0], y_min))

        # Active boxes are the ones where ei_hi is higher than ei_max
        active_boxes_mask = ei_hi > ei_max  # (n,)

        # Update maximum with of active boxes
        w_max = cp.max(bounds_U - bounds_L, axis=0)
        # Split boxes
        bounds_L, bounds_U = split_boxes(bounds_L, bounds_U, epsilon)

        # Update partition count
        partition += 1

    return

def exactbo_loop(X0, bounds, epsilon, gp, f, max_iters, max_partitions):
    # Initialize data
    X_data = X0.copy() # (N,d), initially N=N0

    for iteration in range(max_iters):
        print(f"Iteration {iteration+1}/{max_iters}")
        # Evaluate function at current data points
        y_data = f(X_data) # (N,)

        # Fit Gaussian Process
        gp.fit(X_data, y_data)
        print("GP fitted")

        # Run partitioning to find next point and evaluate it
        X_new = partition_loop(X_data, bounds, epsilon, gp, max_partitions) # (d,)
        y_new = f(X_new) # (1,)
        print(f"Evaluated new point: {X_new} -> {y_new}")

        # Update data
        X_data = np.vstack((X_data, X_new)) # (N+1,d)
        y_data = np.hstack((y_data, y_new)) # (N+1,)

    return X_data, y_data