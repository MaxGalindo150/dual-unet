import numpy as np
from types import SimpleNamespace

def mu_from_d(model: SimpleNamespace, d_est: np.array) -> np.array:
    mu_est = np.zeros(d_est.shape)
    vec_a_est = np.zeros(d_est.shape)
    mu_est[0] = d_est[0]
    
    for iter in range(1, len(d_est)):
        vec_a_est[iter - 1] = np.exp(-mu_est[iter - 1] * model.dz)
        mu_est[iter] = d_est[iter] / np.prod(vec_a_est[:iter])
    return mu_est

def d_from_mu(model: SimpleNamespace, mu_est: np.array) -> np.array:
    d_est = np.zeros(mu_est.shape)
    vec_a_est = np.zeros(mu_est.shape)
    
    d_est[0] = mu_est[0]
    
    for iter in range(1, len(mu_est)):
        vec_a_est[iter - 1] = np.exp(-mu_est[iter - 1] * model.dz)
        d_est[iter] = mu_est[iter] * np.prod(vec_a_est[:iter])
        
    return d_est
    
