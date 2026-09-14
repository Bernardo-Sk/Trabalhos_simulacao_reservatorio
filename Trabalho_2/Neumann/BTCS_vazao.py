import numpy as np
from scipy.linalg import solve

#Gauss-Seidel
def gauss_seidel_vazao(Nx, Nt, p0, qw, rx, dh, A, dx, phi, ct, dt, k, mu, tol=1e-5, max_iter=2000):
    p = np.ones((Nt, Nx)) * p0
    beta = rx * dh
    Cq = (qw * dt) / (A * dx * phi * ct)
    dP_poco = (qw * mu * dx) / (2 * k * A)
    
    for n in range(0, Nt - 1):
        p[n+1, :] = p[n, :].copy() 
        for _ in range(max_iter):
            p_old = p[n+1, :].copy()
            
            p[n+1, 1] = (p[n, 1] - Cq + beta * p[n+1, 2]) / (1 + beta)
            
            for i in range(2, Nx - 2):
                p[n+1, i] = (p[n, i] + beta * p[n+1, i-1] + beta * p[n+1, i+1]) / (1 + 2 * beta)
                
            p[n+1, -2] = (p[n, -2] + (8/3) * beta * p0 + (4/3) * beta * p[n+1, -3]) / (1 + 4 * beta)
            
            if np.max(np.abs(p[n+1, 1:-1] - p_old[1:-1])) < tol:
                break
                
        p[n+1, 0] = p[n+1, 1] - dP_poco
        p[n+1, -1] = p0
        
    return p

#Jacobi
def jacobi_vazao(Nx, Nt, p0, qw, rx, dh, A, dx, phi, ct, dt, k, mu, tol=1e-5, max_iter=2000):
    p = np.ones((Nt, Nx)) * p0
    beta = rx * dh
    Cq = (qw * dt) / (A * dx * phi * ct)
    dP_poco = (qw * mu * dx) / (2 * k * A)
    
    for n in range(0, Nt - 1):
        p[n+1, :] = p[n, :].copy()
        for _ in range(max_iter):
            p_old = p[n+1, :].copy()
            
            p[n+1, 1] = (p[n, 1] - Cq + beta * p_old[2]) / (1 + beta)
            
            for i in range(2, Nx - 2):
                p[n+1, i] = (p[n, i] + beta * p_old[i-1] + beta * p_old[i+1]) / (1 + 2 * beta)
                
            p[n+1, -2] = (p[n, -2] + (8/3) * beta * p0 + (4/3) * beta * p_old[-3]) / (1 + 4 * beta)
            
            if np.max(np.abs(p[n+1, 1:-1] - p_old[1:-1])) < tol:
                break
                
        p[n+1, 0] = p[n+1, 1] - dP_poco
        p[n+1, -1] = p0
        
    return p

#Solver
def solver_vazao(Nx, Nt, p0, qw, rx, dh, A, dx, phi, ct, dt, k, mu):
    p = np.ones((Nt, Nx)) * p0
    beta = rx * dh
    Cq = (qw * dt) / (A * dx * phi * ct)
    dP_poco = (qw * mu * dx) / (2 * k * A)
    
    A_mat = np.zeros((Nx - 2, Nx - 2))
    A_mat[0, 0] = 1 + beta
    A_mat[0, 1] = -beta
    
    for i in range(1, Nx - 3):
        A_mat[i, i-1] = -beta
        A_mat[i, i] = 1 + 2 * beta
        A_mat[i, i+1] = -beta
        
    A_mat[-1, -2] = -4/3 * beta
    A_mat[-1, -1] = 1 + 4 * beta

    for n in range(0, Nt - 1):
        b = p[n, 1:-1].copy()
        b[0] -= Cq
        b[-1] += (8/3) * beta * p0
        
        p[n+1, 1:-1] = solve(A_mat, b)
        p[n+1, 0] = p[n+1, 1] - dP_poco
        p[n+1, -1] = p0
        
    return p