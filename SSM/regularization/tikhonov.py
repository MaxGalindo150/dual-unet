import numpy as np

def tikhonov(H, b,lambdas, dim):
    """
    This function computes the Tikhonov regularization.
    :param H: matrix of the linear model
    :param b: measurements
    :param lambdas: regularization parameter
    :return: estimated signal
    """
    if dim == 1:
        I = np.eye(H.shape[1])
        # Compute the Tikhonov regularization
        d_est_LSQ = np.linalg.inv(H.T @ H + 0.000001*lambdas*I) @ H.T @ b
        return d_est_LSQ
    elif dim == 2:
        I = np.eye(H.shape[1])
        # Compute the Tikhonov regularization
        d_est_LSQ = np.linalg.inv(H.T @ H + lambdas*I) @ H.T @ b
        return d_est_LSQ

