import numpy as np
import matplotlib.pyplot as plt

#Dados
k = 1*(10**(-13)) #permeabilidade
phi = 0.2 #porosidade
mu = 1*(10**(-3)) #viscosidade 
ct = 1*(10**(-9)) #compressibilidade total
L_reservatorio = 100 #comprimento
pw = 15*(10**6) #pressão poço
pe = 25*(10**6) #pressão externa

dh = k / (phi * mu * ct) #Difusividade Hidraulica

#Fazendo fluxo linear pressão
def linear_pressao_pressao(x, t, L, pw, pe, termos=100):
    soma = np.zeros_like(x)
    for n in range (1, termos + 1):
        exp = np.exp(-((n * np.pi / L)**2) *dh *t)
        seno = np.sin(n * np.pi * x / L)
        soma += (1/n) * exp * seno

    p = (pe - pw) * ((x/L) + (2 / np.pi) * soma) + pw
    return p

#Gráfico
x = np.linspace(0, L_reservatorio, 100)
t_dias = [10, 50, 100]
for t in t_dias:
    #t_segs = t * 86400 #passar para segundos por conta da difusividade hidráulica
    f = linear_pressao_pressao(x, t, L_reservatorio, pw, pe)
    p = f/(10**6) #conversão para MPa
    plt.plot(x, p, label=f"Tempo = {t} dias")

plt.title("Fluxo Linear - Pressão Prescrita")
plt.xlabel("Distância (m)")
plt.ylabel("Pressão (MPa)")
plt.legend()
plt.grid(True)
plt.show()