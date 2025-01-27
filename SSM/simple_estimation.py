import numpy as np
from types import SimpleNamespace

def estimation(model: SimpleNamespace, signal: SimpleNamespace) -> SimpleNamespace:

    estimation_results = SimpleNamespace()

    estimation_results.E = np.linalg.inv(model.H.T @ model.H + model.regularization_term * np.eye(model.Nd)) @ model.H.T

    d_est = estimation_results.E @ signal.y
    
    
    mu_est = np.zeros(model.Nd)
    vec_a_est = np.zeros(model.Nd)

    mu_est[0] = d_est[0]

    for iter in range(1, model.Nd):
        vec_a_est[iter - 1] = np.exp(-mu_est[iter - 1] * model.dz)
        mu_est[iter] = d_est[iter] / np.prod(vec_a_est[:iter])

    estimation_results.mu = mu_est

    return estimation_results    
