from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix

def generate_ssm_model(model: SimpleNamespace, probe: SimpleNamespace) -> SimpleNamespace:
    
    I = np.eye(model.Nz)
    
    # Generate matrix D
    D = (1/pow(model.dz ,2))*(-2*np.eye(model.Nz) + np.eye(model.Nz, k=1) + np.eye(model.Nz, k=-1))
    model.D = D

    # Generate matrix M1
    M1 = -1/(pow(probe.c0*model.dt, 2))*I + (probe.tau/(2*model.dt))*D
    model.M1 = M1

    # Generate matrix M2
    model.M2 = D + 2/(pow(probe.c0*model.dt, 2))*I

    # Generate matrix M3
    model.M3 = -1/(pow(probe.c0*model.dt, 2))*I - (probe.tau/(2*model.dt))*D

    M1_inv = np.linalg.inv(M1)

    # Generate matrix M4
    M4 = -M1_inv @ model.M2
    model.M4 = M4

    # Generate matrix M5
    M5 = -M1_inv @ model.M3
    model.M5 = M5

    # Generate matrix A
    zeros_matrix = np.zeros((model.Nz, model.Nz))
    model.A = np.bmat([[model.M4, model.M5],[I, zeros_matrix]])
    model.A[np.abs(model.A) < 1e-40] = 0
    model.A = csr_matrix(model.A)

    # Generate vector b
    d = np.zeros((model.Nd, 1))
    vec_a = np.exp(-probe.mu * model.dz)
    d[0] = probe.mu[0]
    for i in range(1, model.Nd):
        d[i] = probe.mu[i] * np.prod(vec_a[:i-1])
    del vec_a

    # Vector b for the finite difference scheme
    b = -probe.beta*probe.chi/(probe.Cp*model.dt)*d
    b = np.concatenate((np.zeros((model.idx_l-1, 1)), b, np.zeros((model.Nz - model.idx_r + 1, 1))))




    # Vector f for the finite difference scheme
    f = M1_inv @ b
    f[np.abs(f) < 1e-40] = 0

    # Vector g for the finite difference scheme
    model.g = np.concatenate((f, np.zeros((model.Nz, 1))))

    # Vector c for the finite difference scheme
    c = np.zeros((2*model.Nz, 1))
    c[model.idx_l] = 1
    model.c = csr_matrix(c)

    return model




