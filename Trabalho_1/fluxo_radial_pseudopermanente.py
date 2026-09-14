import numpy as np
import matplotlib.pyplot as plt


#Dados
k = 1*(10**(-13)) #permeabilidade
phi = 0.2 #porosidade
mu = 1*(10**(-3)) #viscosidade 
ct = 1*(10**(-9)) #compressibilidade total
p0 = 15*(10**6) #pressão inicial
qw = 0.0002 #vazão prescrita
rw = 0.1 #raio poço
re = 1000 #raio externo
h = 10 #espessura

#Fazendo fluxo radial pseudopermanente
def radial_pseudopermanente(r, t, h, rw, re, p0, qw):
    a = (2 * k *t) / (phi* mu *ct * re**2)
    b = -np.log(r / rw) + 0.5 * (r / re)**2 + np.log(re / rw) - 0.75
    p = p0 - ((qw * mu)/(2 * np.pi * k * h)) * (a + b)
    return p

#Gráfico
r = np.linspace(rw, re, 100)
tempo = (10, 50, 100) 

for t in tempo:
    t_segs = t * 86400 #passar para segundos por conta da difusividade hidráulica
    f = radial_pseudopermanente(r, t_segs, h, rw, re, p0, qw)
    p = f/(10**6) #conversão para MPa
    plt.plot(r, p, label=f"Tempo={t} dias")

plt.title("Fluxo Radial Pseudopermanente")
plt.xlabel("Raio (m)")
plt.ylabel("Pressão (MPa)")
plt.legend()
plt.grid(True)
plt.show()