import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expi

#Dados
k = 1*(10**(-13)) #permeabilidade
phi = 0.2 #porosidade
mu = 1*(10**(-3)) #viscosidade 
ct = 1*(10**(-9)) #compressibilidade total
h = 10 #espessura
p0 = 15*(10**6) #pressão inicial
qw = 0.0002 #vazão prescrita
rw = 0.1 # raio poço
re = 4000 # raio externo

dh = k / (phi * mu * ct) #Difusividade Hidraulica

#Fazendo fluxo radial transiente
def radial_transiente(r, t, h, p0, qw):
    c = (phi * mu * ct * r**2) / (4 * k * t)
    p = p0 + ((qw*mu) / (4 * np.pi * k * h)) * expi(-c)
    return p

#Gráfico
r = np.linspace(rw, re, 100)
tempo = (10, 50, 100) 

for t in tempo:
    t_segs = t * 86400 #passar para segundos por conta da difusividade hidráulica
    f = radial_transiente(r, t_segs, h, p0, qw)
    p = f/(10**6) #conversão para MPa
    plt.plot(r, p, label=f"Tempo={t} dias")

plt.title("Fluxo Radial Transiente")
plt.xlabel("Raio (m)")
plt.ylabel("Pressão (MPa)")
plt.legend()
plt.grid(True)
plt.show()