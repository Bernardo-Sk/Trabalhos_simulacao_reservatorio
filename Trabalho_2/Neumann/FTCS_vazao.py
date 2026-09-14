import numpy as np

def FTCS_vazao(Nx, Nt, p0, qw, rx, dh, A, dx, phi, ct, dt, k, mu):
    p = np.ones((Nt, Nx)) * p0
    p[:, -1] = p0
    
    beta = rx * dh
    if beta > 0.25:
        raise ValueError(f"FTCS instável: beta = {beta:.4f} > 0.25")
        
    Cq = (qw * dt) / (A * dx * phi * ct)
    dP_poco = (qw * mu * dx) / (2 * k * A)
    
    for n in range(0, Nt - 1):
        p[n+1, 1] = p[n, 1] + beta * (p[n, 2] - p[n, 1]) - Cq
        
        for i in range(2, Nx - 2):
            p[n+1, i] = p[n, i] + beta * (p[n, i-1] - 2*p[n, i] + p[n, i+1])
        
        p[n+1, -2] = (4/3)*beta*p[n, -3] + (1 - 4*beta)*p[n, -2] + (8/3)*beta*p0
        
        p[n+1, 0] = p[n+1, 1] - dP_poco
        
    return p