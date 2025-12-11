# Print done
print("Starting - Bound Script")

import sys
import pickle
import tamubo.exactbo as ebo
import cupynumeric as cp
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from legate.timing import time
from partition_cupynumeric import split_boxes

# Print done
print("Modules Loaded")

# Define rbf_k_bounds
def rbf_k_bounds(boxes_bounds_L, boxes_bounds_R, xi, n, d, sigma_f_2, length_scale):
    xi_ext = cp.tile(xi,n)
    
    d_min = cp.maximum(cp.maximum(boxes_bounds_L-xi_ext,xi_ext-boxes_bounds_R),0)
    d_max = cp.maximum(cp.abs(boxes_bounds_L-xi_ext),cp.abs(xi_ext-boxes_bounds_R))

    D_min = cp.linalg.norm(d_min.reshape(n,d),axis=1)
    D_max = cp.linalg.norm(d_max.reshape(n,d),axis=1)

    K_lo = sigma_f_2*cp.exp(-1/(2*length_scale**2)*cp.power(D_max,2))
    K_hi = sigma_f_2*cp.exp(-1/(2*length_scale**2)*cp.power(D_min,2))
    
    
    #K = cp.zeros((n,2))
    #K[:,0] = sigma_f_2*cp.exp(-1/(2*length_scale**2)*cp.power(D_max,2))
    #K[:,1] = sigma_f_2*cp.exp(-1/(2*length_scale**2)*cp.power(D_min,2))

    #return(K)
    return(K_lo,K_hi)

def mu_bounds(alpha,K_lo,K_hi,n,N):
    alphaK_lo = cp.zeros((n,N))
    alphaK_hi = cp.zeros((n,N))
    
    alpha_mask_pos = alpha >= 0

    # when alpha >= 0: hi→hi, lo→lo
    alphaK_hi[:, alpha_mask_pos] = K_hi[:, alpha_mask_pos] * alpha[alpha_mask_pos]
    alphaK_lo[:, alpha_mask_pos] = K_lo[:, alpha_mask_pos] * alpha[alpha_mask_pos]

    # when alpha < 0: hi→lo, lo→hi
    alphaK_lo[:, ~alpha_mask_pos] = K_hi[:, ~alpha_mask_pos] * alpha[~alpha_mask_pos]
    alphaK_hi[:, ~alpha_mask_pos] = K_lo[:, ~alpha_mask_pos] * alpha[~alpha_mask_pos]

    # sum each row
    mu_hi = alphaK_hi.sum(axis=1)  # shape (n,)
    mu_lo = alphaK_lo.sum(axis=1)  # shape (n,)

    return(mu_lo, mu_hi)

def main():
    # Define model
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=0.2, length_scale_bounds=(1e-2, 10.0)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True)

    # Define bounds
    bounds = [[0,1],[0,1]]

    # Define initial box
    init_box = ebo.Box(bounds,True)

    # Function to minimize
    # f(x,y)= \alpha(x^2 + y^2) - \sum_{i=1}^3 A_i \exp \left( -\frac{(x - Cx_i)^2 + (y - Cy_i)^2}{B_i} \right) + D
    def f(X):
        x, y = X[:,0], X[:,1]

        # Parameters
        alpha = 0.1
        A  = np.array([4, 3, 2])
        B  = np.array([0.08, 0.05, 0.02])    # betas
        C  = np.array([[0.9, 0.3 ],      # centers (x1,y1)
                    [0.1 , 0.8],
                    [0.6 , 0.7 ]])
        D = 2
        
        # Compute function value
        val = alpha*(x**2 + y**2)
        for Ai, Bi, (xi, yi) in zip(A, B, C):
            r2 = (x - xi)**2 + (y - yi)**2
            val -= Ai * np.exp(-r2 / Bi)
        val += D
        return val

    # Create initial points and evaluate
    X0 = np.array([[0,0],[1,1],[0.5,0.5],[0,1],[1,0]])
    X0_cp = cp.array(X0.tolist())

    N = X0.shape[0]

    y0 = f(X0)

    # Train model
    gp.fit(X0,y0.ravel())

    # Get parameters
    gp_kernel_params = gp.kernel_.get_params()
    sigma_f_2 = gp_kernel_params['k1__k1__constant_value']
    length_scale = gp_kernel_params['k1__k2__length_scale']
    #sigma_n_2 = gp_kernel_params['k2__noise_level']
    alpha = cp.array(gp.alpha_)

    # Define boxes object
    boxes = ebo.Boxes([init_box])

    # Number of boxes
    n = len(boxes)
    d = init_box.dim
    w = cp.array(init_box.width)

    # Boxes bounds to cupynumeric
    boxes_bounds = boxes.bounds.reshape(n*d,2)
    boxes_bounds_L = cp.array(boxes_bounds[:,0].reshape(n,d))
    boxes_bounds_R = cp.array(boxes_bounds[:,1].reshape(n,d))

    # Loop over different amount of boxes
    ns = []
    times = []
    #Ls = []
    #Rs = []
    #Ks = []
    #K_los = []
    #K_his = []
    mu_los = []
    mu_his = []
    for _ in range(10):
        n = boxes_bounds_L.shape[0]

        # Find kernel bounds (timing it)
        start_time = time()
        #K = cp.zeros((n,2*N))
        K_lo = cp.zeros((n,N))
        K_hi = cp.zeros((n,N))
        for i in range(N):
            xi = X0_cp[i]
            K_lo[:,i], K_hi[:,i] = rbf_k_bounds(boxes_bounds_L.ravel(),boxes_bounds_R.ravel(),xi,n,d,sigma_f_2,length_scale)
            #K[:,2*i:2*(i+1)] = rbf_k_bounds(boxes_bounds_L.ravel(),boxes_bounds_R.ravel(),xi,n,d,sigma_f_2,length_scale)
        mu_lo, mu_hi = mu_bounds(alpha,K_lo,K_hi,n,N)
        end_time = time()

        # Save results
        ns.append(n)
        times.append((end_time-start_time)/1e6)
        #Ks.append(K)
        #Ls.append(boxes_bounds_L)
        #Rs.append(boxes_bounds_R)
        #K_los.append(K_lo)
        #K_his.append(K_hi)
        mu_los.append(mu_lo)
        mu_his.append(mu_hi)

        boxes_bounds_L, boxes_bounds_R = split_boxes(boxes_bounds_L, boxes_bounds_R, w, n, d)

    # Results dictionary
    results = {
        'ns': ns,
        'times': times,
        #'Ls': Ls,
        #'Rs': Rs,
        #'Ks': Ks,
        #'K_los': K_los,
        #'K_his': K_his,
        'mu_los': mu_los,
        'mu_his': mu_his
    }

    # Save results as pickle
    with open("examples/Data/results_bounds_cupynumeric.pkl", "wb") as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()

# Print done
print("Done - Bound Script")