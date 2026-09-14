import numpy as np
import matplotlib.pyplot as plt
import time

import FTCS_vazao as FTCS_vazao
import BTCS_vazao as BTCS_vazao
import Analitico_vazao

# Dados
k = 1*(10**(-13))      # Permeabilidade (m²)
phi = 0.2              # Porosidade
mu = 1*(10**(-3))      # Viscosidade (Pa.s)
ct = 1*(10**(-9))      # Compressibilidade total (Pa^-1)
L = 6000               # Comprimento (m)
p0 = 15*(10**6)        # Pressão inicial (Pa)
qw = 0.0002            # Vazão prescrita (m³/s)
A = 5000               # Área transversal (m²)

dh = k / (phi * mu * ct) # Difusividade Hidráulica (m²/s)

# Discretização
dx = 100 
Nx = int(L / dx) + 2
dt = 4000 
Nt = int((100 * 86400) / dt) + 1 
rx = dt / dx**2

x = np.zeros(Nx)
x[0], x[1] = 0, dx / 2
for i in range(2, Nx - 1):
    x[i] = x[i - 1] + dx
x[-1] = x[-2] + dx / 2

# Métodos
print("Tempo Computacional dos Métodos")

# 1. FTCS Explícito
start = time.time()
p_ftcs = FTCS_vazao.FTCS_vazao(Nx, Nt, p0, qw, rx, dh, A, dx, phi, ct, dt, k, mu)
t_ftcs = time.time() - start
print(f"FTCS (Explícito):       {t_ftcs:.4f} s")

# 2. BTCS Gauss-Seidel
start = time.time()
p_gauss = BTCS_vazao.gauss_seidel_vazao(Nx, Nt, p0, qw, rx, dh, A, dx, phi, ct, dt, k, mu)
t_gauss = time.time() - start
print(f"BTCS (Gauss-Seidel):    {t_gauss:.4f} s")

# 3. BTCS Jacobi
start = time.time()
p_jacobi = BTCS_vazao.jacobi_vazao(Nx, Nt, p0, qw, rx, dh, A, dx, phi, ct, dt, k, mu)
t_jacobi = time.time() - start
print(f"BTCS (Jacobi):          {t_jacobi:.4f} s")

# 4. BTCS Solver
start = time.time()
p_direto = BTCS_vazao.solver_vazao(Nx, Nt, p0, qw, rx, dh, A, dx, phi, ct, dt, k, mu)
t_direto = time.time() - start
print(f"BTCS (Matriz Direta):   {t_direto:.4f} s")

# ------------------------ TABELA DE ERROS (L2 e L-infinito) ------------------------
n_erro = int((100 * 86400) / dt)
t_erro_seg = 100 * 86400

p_an_erro = Analitico_vazao.linear_vazao_prescrita(x, t_erro_seg, L, p0, qw, A) #aplicando analítica

# 1. Norma L2
erro_ftcs_L2 = np.sqrt(np.mean((p_an_erro[1:-1] - p_ftcs[n_erro, 1:-1])**2))
erro_direto_L2 = np.sqrt(np.mean((p_an_erro[1:-1] - p_direto[n_erro, 1:-1])**2))
erro_gauss_L2 = np.sqrt(np.mean((p_an_erro[1:-1] - p_gauss[n_erro, 1:-1])**2))
erro_jacobi_L2 = np.sqrt(np.mean((p_an_erro[1:-1] - p_jacobi[n_erro, 1:-1])**2))

# 2. Norma L∞
erro_ftcs_Linf = np.max(np.abs(p_an_erro[1:-1] - p_ftcs[n_erro, 1:-1]))
erro_direto_Linf = np.max(np.abs(p_an_erro[1:-1] - p_direto[n_erro, 1:-1]))
erro_gauss_Linf = np.max(np.abs(p_an_erro[1:-1] - p_gauss[n_erro, 1:-1]))
erro_jacobi_Linf = np.max(np.abs(p_an_erro[1:-1] - p_jacobi[n_erro, 1:-1]))

tabela_dados = [
    ['FTCS (Explícito)', f"{t_ftcs:.4f}", f"{erro_ftcs_L2 / 1e3:.4f}", f"{erro_ftcs_Linf / 1e3:.4f}"],
    ['BTCS (Solver)', f"{t_direto:.4f}", f"{erro_direto_L2 / 1e3:.4f}", f"{erro_direto_Linf / 1e3:.4f}"],
    ['BTCS (Gauss-Seidel)', f"{t_gauss:.4f}", f"{erro_gauss_L2 / 1e3:.4f}", f"{erro_gauss_Linf / 1e3:.4f}"],
    ['BTCS (Jacobi)', f"{t_jacobi:.4f}", f"{erro_jacobi_L2 / 1e3:.4f}", f"{erro_jacobi_Linf / 1e3:.4f}"]
]
colunas = ['Método Numérico', 'Tempo (s)', 'Erro Global L2 (kPa)', 'Erro Máximo L∞ (kPa)']

print("Erros numéricos e tempo computacional para o Caso 2 (Neumann).")
print(f"{colunas[0]:<22} | {colunas[1]:<10} | {colunas[2]:<20} | {colunas[3]}")
print("-" * 82)
for linha in tabela_dados:
    print(f"{linha[0]:<22} | {linha[1]:<10} | {linha[2]:<20} | {linha[3]}")
print("-" * 82)

#Gráficos
plt.figure(figsize=(12, 7))
tempos_dias = [10, 50, 100]

for i, t_dias in enumerate(tempos_dias):
    t_seg = t_dias * 86400
    n = int(t_seg / dt) 
    
    p_an = Analitico_vazao.linear_vazao_prescrita(x, t_seg, L, p0, qw, A)
    
    # Plot Analítica
    plt.plot(x, p_an / 1e6, 'k-', linewidth=1.5, label='Analítica' if i == 0 else None)
    
    # Plot Métodos
    plt.plot(x, p_ftcs[n] / 1e6, '--', color=f'C{i}', linewidth=2, label=f'FTCS ({t_dias}d)') #Explícita
    plt.plot(x, p_direto[n] / 1e6, 'o', markerfacecolor='none', markeredgecolor=f'C{i}', markersize=8, label='Solver' if i == 0 else None) #Solver
    plt.plot(x, p_gauss[n] / 1e6, 'x', color=f'C{i}', markersize=6, label='Gauss' if i == 0 else None) #Gauss-Seidel
    plt.plot(x, p_jacobi[n] / 1e6, '^', markerfacecolor='none', markeredgecolor=f'C{i}', markersize=7, label='Jacobi' if i == 0 else None) #Jacobi

plt.title("Simuladores FDM: Explícito vs Implícitos vs Analítica (Neumann)")
plt.xlabel('Distância x (m)')
plt.ylabel('Pressão (MPa)')

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()