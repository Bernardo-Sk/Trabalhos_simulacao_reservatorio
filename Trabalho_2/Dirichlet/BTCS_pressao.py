import numpy as np
from scipy.linalg import solve

#Gauss Seidel
def gauss_seidel(Nx, Nt, p0, pw, rx, dh, tol=1e-5, max_iter=1000):
    p = np.ones((Nt, Nx)) * p0
    p[:, 0] = pw
    p[:, -1] = p0
    beta = rx * dh
    
    for n in range(0, Nt - 1):
        p[n+1, :] = p[n, :].copy()
        for _ in range(max_iter):
            p_old = p[n+1, :].copy()
            
            p[n+1, 1] = (p[n, 1] + (8/3)*beta*pw + (4/3)*beta*p[n+1, 2]) / (1 + 4*beta)
            
            for i in range(2, Nx - 2):
                p[n+1, i] = (p[n, i] + beta*p[n+1, i-1] + beta*p[n+1, i+1]) / (1 + 2*beta)
                
            p[n+1, -2] = (p[n, -2] + (8/3)*beta*p0 + (4/3)*beta*p[n+1, -3]) / (1 + 4*beta)
            
            if np.max(np.abs(p[n+1, :] - p_old)) < tol:
                break
    return p

#Jacobi
def jacobi(Nx, Nt, p0, pw, rx, dh, tol=1e-5, max_iter=1000):
    p = np.ones((Nt, Nx)) * p0
    p[:, 0] = pw
    p[:, -1] = p0
    beta = rx * dh
    
    for n in range(0, Nt - 1):
        p[n+1, :] = p[n, :].copy()
        for _ in range(max_iter):
            p_old = p[n+1, :].copy()

            p[n+1, 1] = (p[n, 1] + (8/3)*beta*pw + (4/3)*beta*p_old[2]) / (1 + 4*beta)
            
            for i in range(2, Nx - 2):
                p[n+1, i] = (p[n, i] + beta*p_old[i-1] + beta*p_old[i+1]) / (1 + 2*beta)
                
            p[n+1, -2] = (p[n, -2] + (8/3)*beta*p0 + (4/3)*beta*p_old[-3]) / (1 + 4*beta)
            
            if np.max(np.abs(p[n+1, :] - p_old)) < tol:
                break
    return p

#Solver
def solver(Nx, Nt, p0, pw, rx, dh):
    p = np.ones((Nt, Nx)) * p0
    p[:, 0] = pw
    p[:, -1] = p0
    beta = rx * dh

    A = np.zeros((Nx - 2, Nx - 2))
    
    A[0, 0] = 1 + 4 * beta
    A[0, 1] = -4/3 * beta
    
    for i in range(1, Nx - 3):
        A[i, i-1] = -beta
        A[i, i] = 1 + 2 * beta
        A[i, i+1] = -beta
        
    A[-1, -2] = -4/3 * beta
    A[-1, -1] = 1 + 4 * beta

    for n in range(0, Nt - 1):
        b = p[n, 1:-1].copy()
        b[0] += (8/3) * beta * pw
        b[-1] += (8/3) * beta * p0
        
        p[n+1, 1:-1] = solve(A, b)
        
    return p