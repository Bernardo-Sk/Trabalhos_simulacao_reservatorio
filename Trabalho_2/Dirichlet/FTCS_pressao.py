import numpy as np

def FTCS(Nx, Nt, p0, pw, rx, dh):
    p = np.ones((Nt, Nx)) * p0
    p[:, 0] = pw
    p[:, -1] = p0
    
    if rx * dh > 0.25:
        raise ValueError(f"FTCS instável: rx*dh = {rx*dh:.4f} > 0.25")
    
    for n in range(0, Nt - 1):
        p[n+1, 1] = (8/3) * rx * dh * pw + (1 - 4 * rx * dh) * p[n, 1] + (4/3) * rx * dh * p[n, 2]
        
        for i in range(2, Nx - 2):
            p[n+1, i] = rx * dh * p[n, i-1] + (1 - 2 * rx * dh) * p[n, i] + rx * dh * p[n, i+1]
            
        p[n+1, -2] = (4/3) * rx * dh * p[n, -3] + (1 - 4 * rx * dh) * p[n, -2] + (8/3) * rx * dh * p0
        
    return p