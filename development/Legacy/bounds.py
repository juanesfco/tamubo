from __future__ import annotations
import numpy as np
from .partition import Box
from .interval_arithmetics import Bounds, prod_bound_scalar, add_bounds, sub_bounds, prod_bounds, square_bounds, sqrt_bounds, forward_solve_bounds
from scipy.stats import norm
from typing import Tuple

Array = np.ndarray

def rbf_k_bounds(L: Array, R: Array, xi: Array, model) -> Tuple[float, float]:
  """
  Bounds [K_min, K_max] for k(x, xi) of an RBF with x in the box [L, R].
  Works for sklearn kernels like ConstantKernel*RBF and (+ WhiteKernel).
  """
  # Ensure inputs are 1D arrays of same length
  L = np.asarray(L, dtype=float).ravel()
  R = np.asarray(R, dtype=float).ravel()
  xi = np.asarray(xi, dtype=float).ravel()
  assert L.shape == R.shape == xi.shape, "L, R, xi must have same shape"
  assert np.all(L <= R), "Each component must satisfy L[j] <= R[j]"

  # Compute dmin and dmax for each dimension
  dmins = np.maximum(np.maximum(L-xi, xi-R), 0)  # Distance to box if outside, else 0
  dmaxs = np.maximum(np.abs(L - xi), np.abs(R - xi))  # Max distance to box corners

  # Bounds for D_i
  Dmin = np.linalg.norm(dmins)  # Minimum distance to box
  Dmax = np.linalg.norm(dmaxs)  # Maximum distance to box corners
  
  # Calculate Kmin and Kmax using kernel from gp
  Kall = model.kernel_([[Dmin],[Dmax]],[[0],[0]])
  KmaxGP, KminGP = Kall.diagonal()
  
  return float(KminGP), float(KmaxGP)

def mu_bounds(box: Box, model) -> Tuple[Bounds, Bounds]:
  """
  Explicit IA μ bounds for x on a Box using:
    μ(x)=k(x)^T α, α = L^{-T} \\ (L \\ y),
  Inputs:
    box    : Box
    model : Trined GP surrogate
  Returns:
    Bounds(μ_lo, μ_hi), Bounds([k_lo_i], [k_hi_i])
  """
  X = np.asarray(model.X_train_, dtype=float)
  n = X.shape[0]
  alpha = model.alpha_.ravel()
  L, R = box.bounds[:,0], box.bounds[:,1]

  # Componentwise kernel intervals
  k = Bounds(np.empty(n, dtype=float), np.empty(n, dtype=float))
  for i, xi in enumerate(X):
    k.lo[i], k.hi[i] = rbf_k_bounds(L, R, xi, model)

  # μ bounds: sum of α_i * K_i with interval multiplication
  mu = Bounds(0, 0)
  for i in range(n):
    alpha_i = alpha[i]
    k_i = Bounds(k.lo[i], k.hi[i])
    mu = add_bounds(mu, prod_bound_scalar(alpha_i, k_i))

  # To undo normalization
  y_train_std = np.asarray(model._y_train_std).item()
  y_train_mean = np.asarray(model._y_train_mean).item()

  mu.lo = y_train_std*float(mu.lo) + y_train_mean
  mu.hi = y_train_std*float(mu.hi) + y_train_mean

  return mu, k

def sigma_bounds(k: Bounds, model) -> Bounds:
  """
  Given L = cholesky(K + σ_n^2 I) and per-component kernel 
  intervals k = Bounds([k_lo_i],[k_hi_i]) (nonnegative),
  bound v = Bounds([v_lo_j], [v_hi_j]) for the solution of L v = k, and then
  bound Q = ||v||^2, then σ^2 = σ_f^2 - Q, 
  and finally σ = Bounds(sqrt(max(0,σ^2_lo)), sqrt(max(0,σ^2_hi))).
  """
  # Extract σ_f^2 from gp
  sigma_f2 = model.kernel_([0]).item()

  # Extract L = cholesky(K + σ_n^2 I) from gp
  L_ = model.L_
  n = L_.shape[0]

  # Calculate bounds v = Bounds([v_lo_j], [v_hi_j]) for the solution of L v = k
  v = forward_solve_bounds(L_, k)

  # Q bounds = sum of squares intervals (IA addition)
  Q = Bounds(0, 0)
  for i in range(n):
    v_i = Bounds(v.lo[i], v.hi[i])
    s = square_bounds(v_i)
    Q = add_bounds(Q, s)

  # σ^2 bounds: [max(0, σ_f^2 - Q_hi), max(0, σ_f^2 - Q_lo)]
  sig2 = Bounds(max(0.0, sigma_f2 - Q.hi), max(0.0, sigma_f2 - Q.lo))

  # σ interval by monotonicity of sqrt
  sig = sqrt_bounds(sig2)

  # To undo normalization
  y_train_std = np.asarray(model._y_train_std).item()

  return Bounds(float(sig.lo)*y_train_std, float(sig.hi)*y_train_std)

def ei_bounds_from_mu_sigma(mu: Bounds, sig: Bounds, model) -> Bounds:
  """
  Explicit IA bounds for EI on an interval using:
    EI = N * norm.cdf(Z) + σ * norm.pdf(Z),  with N=f_min - μ,  Z=N/σ.
  Inputs:
    mu    : Bounds(μ_lo, μ_hi)
    sig   : Bounds(σ_lo, σ_hi) with σ_lo >= 0
    model : Trined GP surrogate
  Returns:
    Bounds(EI_lo, EI_hi)
  """
  # Include 0 in EI bounds flag
  include_0_flag = False
  # Extract f_min from model
  y = np.asarray(model.y_train_, dtype=float).ravel()
  y_min = np.min(y)
  # To undo normalization
  y_train_std = np.asarray(model._y_train_std).item()
  y_train_mean = np.asarray(model._y_train_mean).item()
  f_min = y_train_std*y_min + y_train_mean
  # Bounds for N = f_min - μ
  N = Bounds(f_min - mu.hi, f_min - mu.lo)
  
  # This can be improved
  # If σ_hi == 0, EI must be 0, if σ_lo == 0
  # use a safe enclosure: choose a pad for σ_lo
  # and keep in mind EI bounds must include 0.
  if sig.hi == 0.0:
    return Bounds(0,0)
  else:
    if sig.lo == 0:
      pad = 1e-5
      sig.lo = pad
      include_0_flag = True
  
  # Otherwise, proceed with explicit IA through Z, norm.cdf, norm.pdf
  # J = 1/σ ∈ [1/σ_hi, 1/σ_lo]
  J = Bounds(1/sig.hi, 1/sig.lo)
  # Z = N * J, full interval product
  Z = prod_bounds(N, J)
  # Phi = norm.cdf(Z) bound (monotone)
  Phi = Bounds(norm.cdf(Z.lo), norm.cdf(Z.hi))
  # phi = norm.pdf(Z) bound (unimodal, symmetric)
  phi_candidates = [norm.pdf(Z.lo), norm.pdf(Z.hi)]
  if Z.lo <= 0.0 <= Z.hi:
      phi_candidates.append(norm.pdf(0))
  phi = Bounds(min(phi_candidates), max(phi_candidates))
  # U = N * Phi
  U = prod_bounds(N, Phi)
  # V = σ * phi
  V = prod_bounds(sig, phi)
  # EI = U + V 
  EI = add_bounds(U, V)
  
  if include_0_flag:
    return Bounds(min(0, EI.lo), max(0, EI.hi)) # Ensure zero is in bounds
  else:
    return EI

def ei_bounds(box: Box, model) -> Bounds:
  """
  Compute EI interval [EI_lo, EI_hi] on x in box using a gp model:
    1) kernel intervals -> 2) μ interval -> 3) σ interval -> 4) EI interval
  Returns:
    Bounds object
  """
  mu, k = mu_bounds(box, model)
  sig = sigma_bounds(k, model)
  EI = ei_bounds_from_mu_sigma(mu, sig, model)
  return EI