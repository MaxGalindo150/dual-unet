import numpy as np

def lcfun(lambdas, s, beta, xi):
    """
    Esta función calcula la curvatura de la curva L.
    """
    phi = np.zeros(np.size(lambdas))
    dphi = np.zeros(np.size(lambdas))
    psi = np.zeros(np.size(lambdas))
    dpsi = np.zeros(np.size(lambdas))
    eta = np.zeros(np.size(lambdas))
    rho = np.zeros(np.size(lambdas))

    if np.size(beta) > np.size(s):
        LS = True
        rhoLS2 = beta[-1] ** 2
        beta = beta[0:-1]
    else:
        LS = False

    for i in range(np.size(lambdas)):
        f = s ** 2 / (s ** 2 + lambdas[i] ** 2)
        cf = 1 - f
        eta[i] = np.linalg.norm(f * xi)
        rho[i] = np.linalg.norm(cf * beta)
        f1 = -2 * f * cf / lambdas[i]  # Corregido para incluir `f`
        f2 = -f1 * (3-4*f) / lambdas[i]
        phi[i] = np.sum(f * f1 * np.abs(xi) ** 2)
        psi[i] = np.sum(cf * f1 * np.abs(beta) ** 2)  # Corregido `xi` a `beta`
        dphi[i] = np.sum((f1 ** 2 + f * f2) * np.abs(xi) ** 2)
        dpsi[i] = np.sum((-f1 ** 2 + cf * f2) * np.abs(beta) ** 2)

    if LS: 
        rho = np.sqrt(rho ** 2 + rhoLS2)

    # Compute the first and second derivatives of rho and eta
    deta = phi / eta
    drho = -psi / rho
    ddeta = dphi / eta - deta * (deta / eta)
    ddrho = -dpsi / rho - drho * (drho / rho)

    # Convert to derivatives of log(rho) and log(eta)
    dlogeta = deta / eta
    dlogrho = drho / rho 
    ddlogeta = ddeta / eta - (dlogeta) ** 2 
    ddlogrho = ddrho / rho - (dlogrho) ** 2
    
    # Let g = curvature
    g = - (dlogrho * ddlogeta - ddlogrho * dlogeta) \
            / (dlogeta ** 2 + dlogrho ** 2) ** (1.5)

    return g

