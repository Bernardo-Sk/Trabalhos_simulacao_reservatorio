import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

#Dados
k = 1*(10**(-13)) #permeabilidade
phi = 0.2 #porosidade
mu = 1*(10**(-3)) #viscosidade 
ct = 1*(10**(-9)) #compressibilidade total
L_reservatorio = 6000 #comprimento
p0 = 15*(10**6) #pressão inicial
qw = 0.0002 #vazão prescrita
A = 5000 #área transversal

dh = k / (phi * mu * ct) #Difusividade Hidraulica

#Fazendo fluxo linear vazão
def linear_vazao_prescrita(x, t, L, p0, qw, A):
    a = np.sqrt((4 * dh * t) / (np.pi * L**2)) * np.exp(-(x**2) / (4 *dh * t))
    b = (x / L) * erfc(x / np.sqrt(4 * dh * t))
    p = p0 - ((qw * mu * L) / (k * A)) * (a-b)
    return p

#Gráfico
x = np.linspace(0, L_reservatorio, 100)
t_dias = [10, 50, 100]
for t in t_dias:
    t_segs = t * 86400 #passar para segundos por conta da difusividade hidráulica
    f = linear_vazao_prescrita(x, t_segs, L_reservatorio, p0, qw, A)
    p = f/(10**6) #conversão para MPa
    plt.plot(x, p, label=f't = {t} dias')

plt.title('Fluxo Linear Vazão Prescrita')
plt.xlabel('Distância (m)')
plt.ylabel('Pressão (MPa)')
plt.legend()
plt.grid(True)
plt.show()