from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import convolution_matrix

def generate_linear_model(model: SimpleNamespace, signal: SimpleNamespace, probe: SimpleNamespace) -> SimpleNamespace:
    # Generate matrix M6
        imp = np.zeros((model.Ny, 2*model.Nz))
        imp[1, :] = model.c.T.toarray()
        for iter in range(2, model.Ny):
            imp[iter, :] = imp[iter-1, :]@model.A
        imp2 = np.vstack((np.zeros((1, 2*model.Nz)), imp[:-1, :]))
        M6 = imp - imp2
        M6[np.abs(M6) < 1e-40] = 0
        model.M6 = csr_matrix(M6)
        model.imp_uy = np.matmul(imp, model.g)
        model.imp_iy = np.matmul(M6, model.g)
        del imp
        del imp2

        # Generate convolution matrix C(i)
        input_signal = np.concatenate((signal.i, np.zeros((model.Ny - signal.Ni, 1))))
        C = convolution_matrix(input_signal.flatten(), model.Ny,mode='full')
        model.C = csr_matrix(C[:model.Ny, :])

        # Generate matrix H
        H = (model.C@model.M6)@(np.vstack((np.eye(model.Nz), np.zeros((model.Nz, model.Nz)))))
        inv_M1 = np.linalg.inv(model.M1)
        #print(f'inv_M1: {inv_M1.shape}')
        #inv_M1 = np.delete(inv_M1, np.s_[self.idx_r:], axis=1)
        #inv_M1 = np.delete(inv_M1, np.s_[:self.idx_l-1], axis=1)
        #print(f'inv_M1: {inv_M1.shape}')
        H = np.matmul(H, inv_M1)
        model.H = -probe.beta * probe.chi / (probe.Cp * model.dt) * H

        # Check if H has full rank
        if np.linalg.matrix_rank(model.H) < model.Nd:
            print('Matrix H has not full rank. Consider shortening vector d or regularization!')
            #exit()
        return model
