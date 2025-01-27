import numpy as np
import matplotlib.pyplot as plt
from SSM.regularization.l_corner import l_corner
def l_curve(U, sm, b, method, L, V):
    """
    This function computes the L curve method to find the regularization parameter
    for Tikhonov regularization.
    """
    # Initialize the variables
    n_points = 200
    smin_ratio = 16*np.finfo(float).eps
    
    m, n = U.shape
    p, ps = sm.shape

    beta = U.conj().T @ b
    beta2 = np.linalg.norm(b) ** 2 - np.linalg.norm(beta) ** 2

    if ps == 1:
        s = sm
        beta = beta[0:p]
    else:
        s = sm[p-1::-1,0] / sm[p-1::-1,0]
        beta = beta[p-1::-1]
     
    xi = beta[0:p] / s
    xi[np.isinf(xi)] = 0

    if method == "Tikh" or method == "tikh":

        # eta = ||x||^2
        eta = np.zeros((n_points,1))

        # rho = ||Ax - b||^2
        rho = np.zeros((n_points,1))

        # reg_param = regularization parameter
        reg_param = np.zeros((n_points,1))

        s2 = s ** 2
        
        reg_param[n_points-1] = max(s[p-1], s[0] * smin_ratio)
        ratio = (s[0] / reg_param[n_points-1]) ** (1 / (n_points - 1))
        
        for i in range(n_points-2,-1,-1):
            reg_param[i] = reg_param[i+1] * ratio

        for i in range(n_points):
            f = s2 / (s2 + reg_param[i] ** 2)

            eta[i] = np.linalg.norm(f * xi)

            rho[i] = np.linalg.norm((1 - f) * beta[0:p])

        if (m > n) and (beta2 > 0):
            rho = np.sqrt(rho ** 2 + beta2)

    
    reg_corner, rho_c, eta_c = l_corner(rho, eta, reg_param, U, s, b, "Tikh")
    # print(f"(rho_c, eta_c): ({rho_c}, {eta_c})")
    return  rho, eta, reg_corner, rho_c, eta_c, reg_param


def plot_lc(rho, eta, reg_corner, rho_c, eta_c):
    """
    This function plots the L curve and marks the corner found with a more professional and aesthetic format.
    """
    plt.figure(figsize=(8, 6))  # Set a larger figure size for better readability
    
    # Plot the L curve
    plt.plot(rho, eta, "o-", label="L Curve", linewidth=2, markersize=6, color="#1f77b4")
    
    # Use logarithmic scale
    plt.xscale("log")
    plt.yscale("log")
    
    # Enhance axis labels and title
    plt.xlabel(r"$\|Ax - b\|$", fontsize=14, labelpad=10)
    plt.ylabel(r"$\|x\|$", fontsize=14, labelpad=10)
    plt.title("L-Curve", fontsize=16, weight='bold', pad=15)
    
    # Add grid for better visual guidance
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Highlight the corner point
    plt.plot(rho_c, eta_c, 'ro', label=f'Corner: {reg_corner}', markersize=8)
    
    # Annotate the corner point
    plt.annotate(f'Corner: {reg_corner}', (rho_c, eta_c),
                 textcoords="offset points", xytext=(10, -10), ha='center',
                 color='red', fontsize=12)
    
    # Add legend with better placement
    plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True, fancybox=True)
    
    # Save the figure with a professional format
    plt.savefig("img/L_curve_professional.png", dpi=300, bbox_inches='tight')
    plt.close()
        
