# Print starting
#print("Starting - Loops Script")

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
    bounds_L = cp.asarray([bounds[:,0]], dtype=cp.float64) # (n,d)
    bounds_U = cp.asarray([bounds[:,1]], dtype=cp.float64) # (n,d)

    # GP hyperparameters
    gp_kernel_params = gp.kernel_.get_params()
    sigma_f_2 = gp_kernel_params['k1__k1__constant_value']
    length_scale = gp_kernel_params['k1__k2__length_scale']
    alpha = cp.array(gp.alpha_)
    y_train_std = gp._y_train_std
    y_train_mean = gp._y_train_mean
    L = cp.array(gp.L_)
    y_min = cp.min(gp.y_train_)
    y_min_unscaled = y_min * y_train_std + y_train_mean

    # Partition parameters
    N = X_data.shape[0]  # Number of data points
    d = bounds.shape[0]  # Number of dimensions
    w = bounds_U[0] - bounds_L[0]  # Bounds with per dimension (d,)
    partition = 0
    w_max = w.copy()
    ei_max = 0

    while partition < max_partitions and cp.any(w_max > epsilon):
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
        max_ei_hi_box_L = bounds_L[idx_max_ei_hi,:]  # (d,)
        max_ei_hi_box_U = bounds_U[idx_max_ei_hi,:]  # (d,)
        max_ei_hi_box_center = (max_ei_hi_box_L + max_ei_hi_box_U) / 2.0  # (d,)
        mu_pred, sigma_pred = gp.predict(np.array(max_ei_hi_box_center).reshape(1,-1), return_std=True)
        ei_max = max(ei_max,expected_improvement(mu_pred[0], sigma_pred[0], y_min_unscaled))

        # Active boxes are the ones where ei_hi is higher than ei_max
        active_boxes_mask = ei_hi > ei_max  # (n,)
        if not bool(cp.any(active_boxes_mask)):
            idx_max = cp.argmax(ei_hi)
            active_boxes_mask = cp.zeros(n, dtype=bool)
            active_boxes_mask[idx_max] = True

        # Update maximum width of active boxes
        w_max = cp.max(bounds_U[active_boxes_mask] - bounds_L[active_boxes_mask], axis=0)
        
        # Update partition count
        partition += 1

        # Split active boxes (don't if its the last partition)
        if partition < max_partitions and bool(cp.any(w_max > epsilon)):
            bounds_L, bounds_U = split_boxes(bounds_L, bounds_U, active_boxes_mask, w, n, d)

        # Print status
        print(f" Partition {partition}/{max_partitions}, Boxes: {n}, Active: {cp.sum(active_boxes_mask)}, Max EI: {ei_max:.6f}, Max Width: {w_max}")

    # Check EI in the center of the active boxes and return the best point
    bound_U_active = bounds_U[active_boxes_mask]  # (m,d)
    bound_L_active = bounds_L[active_boxes_mask]  # (m,d)
    center_active = (bound_L_active + bound_U_active) / 2.0  # (m,d)
    mu_active, sigma_active = gp.predict(np.array(center_active), return_std=True) # (m,) both
    ei_active = expected_improvement(cp.array(mu_active), cp.array(sigma_active), y_min_unscaled)  # (m,)
    idx_best = cp.argmax(ei_active)
    best_x = center_active[idx_best]

    return best_x

def exactbo_loop(X0, bounds, epsilon, gp, f, max_iters, max_partitions):
    # Initialize data
    X_data = X0.copy() # (N,d), initially N=N0

    for iteration in range(max_iters):
        print(f"Iteration {iteration+1}/{max_iters}")
        # Evaluate function at current data points
        y_data = f(X_data) # (N,)
        print(f"Current training data: X: {X_data}, y: {y_data}")

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
