import numpy as np
from numpy._core.defchararray import lower
from SSM.regularization.lcfun import lcfun
from scipy.optimize import minimize_scalar
def l_corner(rho, eta, reg_param, U, s, b, method):
    """"
    This function computes the corner of the L curve.
    """

    # Ensure that rho and eta are column vectors

    rho = rho.reshape(-1, 1)
    eta = eta.reshape(-1, 1)
    
    # Initialize
    p, ps = s.shape
    m, n = U.shape

    beta = U.conj().T @ b

    b0 = (b - U @ beta).reshape(-1, 1)
    
    if ps == 2:
        s = s[p-1::-1, 0] / s[p-1::-1, 1]
        beta = beta[p-1::-1]

    xi = beta[0:p] / s

    if m > n:
        beta = np.concatenate((beta, [[np.linalg.norm(b0)]]))

    if method == "Tikh" or method == "tikh":
        # The L-curve is differentiable; computation of curvature
        # in log log scale is easy.

        # Compute g = - curvature of L-curve
        g = lcfun(reg_param, s, beta, xi)
        
        # Find the corner. If the curvature is negative everywhere,
        # then define the leftmost point of the L-curve as the corner.
        gi = np.argmin(g)

        lower_bound = reg_param[min(gi + 1, len(g) -1)]
        upper_bound = reg_param[max(gi - 1, 0)]

        result = minimize_scalar(lcfun, bounds = (lower_bound, upper_bound), args = (s, beta, xi), method = 'bounded')
        reg_c = result.x

        kappa_max = -lcfun(reg_c, s, beta, xi)

        if kappa_max < 0:
            lr = np.size(rho)
            reg_c = reg_param[lr-1]
            rho_c = rho[lr-1]
            eta_c = eta[lr]
        else:
            f = s ** 2 / (s ** 2 + reg_c ** 2)
            eta_c = np.linalg.norm(f * xi)
            rho_c = np.linalg.norm((1 - f) * beta[0:np.size(f)])
            if m > n:
                rho_c = np.sqrt(rho_c ** 2 + np.linalg.norm(b0) ** 2)

    return reg_c, rho_c, eta_c

