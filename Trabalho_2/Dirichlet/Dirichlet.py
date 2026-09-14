import numpy as np
import matplotlib.pyplot as plt
import time

import FTCS_pressao as FTCS_pressao
import BTCS_pressao as BTCS_pressao 
import Analitico_pressao

# Dados
k = 1*(10**(-13))        # permeabilidade (m²)
phi = 0.2                # porosidade
mu = 1*(10**(-3))        # viscosidade (Pa.s)
ct = 1*(10**(-9))        # compressibilidade total (Pa^-1)
L_reservatorio = 100     # comprimento (m)
pw = 15*(10**6)          # pressão poço (Pa)
pe = 25*(10**6)          # pressão externa e inicial (Pa)

dh = k / (phi * mu * ct) # Difusividade Hidráulica (m²/s)

# Discretização
dx = 10
Nx = int(L_reservatorio / dx) + 2
dt = 10  # Tamanho do passo de tempo em segs
Nt = int((2 * 3600) / dt) + 1 # +1 para o tempo zero e evitar limites de matriz
rx = dt / dx**2

x = np.zeros(Nx)
x[0], x[1] = 0, dx / 2
for i in range(2, Nx - 1):
    x[i] = x[i - 1] + dx
x[-1] = x[-2] + dx / 2

# Métodos
print("Tempo Computacional dos Métodos")

#1. FTCS Explícito
start = time.time()
p_ftcs = FTCS_pressao.FTCS(Nx, Nt, pe, pw, rx, dh)
t_ftcs = time.time() - start
print(f"FTCS (Explícito):       {t_ftcs:.4f} s")

#2. BTCS Gauss-Seidel
start = time.time()
p_gauss = BTCS_pressao.gauss_seidel(Nx, Nt, pe, pw, rx, dh)
t_gauss = time.time() - start
print(f"BTCS (Gauss-Seidel):    {t_gauss:.4f} s")

#3. BTCS Jacobi
start = time.time()
p_jacobi = BTCS_pressao.jacobi(Nx, Nt, pe, pw, rx, dh)
t_jacobi = time.time() - start
print(f"BTCS (Jacobi):          {t_jacobi:.4f} s")

#4. BTCS Solver
start = time.time()
p_solver = BTCS_pressao.solver(Nx, Nt, pe, pw, rx, dh)
t_solver = time.time() - start
print(f"BTCS (Solver):          {t_solver:.4f} s")

#Erros
t_erro = 3600
n_erro = int(t_erro / dt)

p_an_erro = Analitico_pressao.linear_pressao_pressao(x, t_erro, L_reservatorio, pw, pe) #aplicando analítica

#1. Norma L2
erro_ftcs_L2 = np.sqrt(np.mean((p_an_erro[1:-1] - p_ftcs[n_erro, 1:-1])**2))
erro_solver_L2 = np.sqrt(np.mean((p_an_erro[1:-1] - p_solver[n_erro, 1:-1])**2))
erro_gauss_L2 = np.sqrt(np.mean((p_an_erro[1:-1] - p_gauss[n_erro, 1:-1])**2))
erro_jacobi_L2 = np.sqrt(np.mean((p_an_erro[1:-1] - p_jacobi[n_erro, 1:-1])**2))

#2. Norma L∞
erro_ftcs_Linf = np.max(np.abs(p_an_erro[1:-1] - p_ftcs[n_erro, 1:-1]))
erro_solver_Linf = np.max(np.abs(p_an_erro[1:-1] - p_solver[n_erro, 1:-1]))
erro_gauss_Linf = np.max(np.abs(p_an_erro[1:-1] - p_gauss[n_erro, 1:-1]))
erro_jacobi_Linf = np.max(np.abs(p_an_erro[1:-1] - p_jacobi[n_erro, 1:-1]))

tabela_dados = [
    ['FTCS (Explícito)', f"{t_ftcs:.4f}", f"{erro_ftcs_L2 / 1e3:.4f}", f"{erro_ftcs_Linf / 1e3:.4f}"],
    ['BTCS (Solver)', f"{t_solver:.4f}", f"{erro_solver_L2 / 1e3:.4f}", f"{erro_solver_Linf / 1e3:.4f}"],
    ['BTCS (Gauss-Seidel)', f"{t_gauss:.4f}", f"{erro_gauss_L2 / 1e3:.4f}", f"{erro_gauss_Linf / 1e3:.4f}"],
    ['BTCS (Jacobi)', f"{t_jacobi:.4f}", f"{erro_jacobi_L2 / 1e3:.4f}", f"{erro_jacobi_Linf / 1e3:.4f}"]
]
colunas = ['Método Numérico', 'Tempo (s)', 'Erro Global L2 (kPa)', 'Erro Máximo L∞ (kPa)']

print("Erros numéricos e tempo computacional para o Caso 1 (Dirichlet).}")
print(f"{colunas[0]:<22} | {colunas[1]:<10} | {colunas[2]:<20} | {colunas[3]}")
print("-" * 82)
for linha in tabela_dados:
    print(f"{linha[0]:<22} | {linha[1]:<10} | {linha[2]:<20} | {linha[3]}")
print("-" * 82)

#Gráficos
plt.figure(figsize=(12, 7))

for i in range(0, Nt, Nt // 5):
    t_seg = i * dt
    tempo_horas = t_seg / 3600
    
    #Plot Analítica
    if t_seg == 0:
        p_an = np.ones_like(x) * pe
        p_an[0] = pw
    else:
        p_an = Analitico_pressao.linear_pressao_pressao(x, t_seg, L_reservatorio, pw, pe)

    #Plot Analítica    
    plt.plot(x, p_an / 1e6, 'k-', linewidth=1.5, label='Analítica' if i == 0 else "")
    
    #Plot Métodos
    plt.plot(x, p_ftcs[i] / 1e6, '--', color=f'C{i}', linewidth=2, label=f'FTCS (t={tempo_horas:.2f}h)') #Explícito
    plt.plot(x, p_solver[i] / 1e6, 'o', markerfacecolor='none', markeredgecolor=f'C{i}', markersize=8, label='Solver' if i == 0 else None) #Solver
    plt.plot(x, p_gauss[i] / 1e6, 'x', color=f'C{i}', markersize=6, label='Gauss' if i == 0 else None) #Gauss-Seidel
    plt.plot(x, p_jacobi[i] / 1e6, '^', markerfacecolor='none', markeredgecolor=f'C{i}', markersize=7, label='Jacobi' if i == 0 else None) #Jacobi
    
plt.title('Simuladores FDM: Explícito vs Implícitos vs Analítica (Dirichlet)')
plt.xlabel('Distância x (m)')
plt.ylabel('Pressão (MPa)')

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()