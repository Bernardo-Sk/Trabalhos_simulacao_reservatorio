import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# Dados do reservatório:
comprimento_x = 1000.0       # m
comprimento_y = 1000.0       # m
espessura = 10.0             # m
numero_celulas_x = 30
numero_celulas_y = 30

# Propriedades da rocha e do fluido:
porosidade = 0.2
viscosidade = 1e-3           # Pa.s
compressibilidade_total = 1e-9  # 1/Pa
pressao_inicial = 25e6       # Pa
raio_poco = 0.1              # m

# Permeabilidade heterogênea:
permeabilidade_mediana = 500.0  # mD
desvio_log = 1.0
semente = 42

# Tempo da simulação:
tempo_total = 365.0          # dias
passo_tempo = 1.0            # dia
dias_dos_graficos = [0, 15, 30, 90, 365]


# Casos simulados:
casos = [
    {
        'nome': 'Caso 1 - Meio heterogêneo',
        'pressao_produtor': 15e6,
        'fator_permeabilidade': 1.0,
        'produtores': [(500.0, 500.0)],
        'injetores': [],
        'falha': None,
        'dias_dos_graficos': [0, 1, 3, 7, 30],
    },
    {
        'nome': 'Caso 2 - Falha selante',
        'pressao_produtor': 15e6,
        'fator_permeabilidade': 1.0,
        'produtores': [(50.0, 50.0)],
        'injetores': [],
        'falha': (100.0, 0.0, 700.0),
        'dias_dos_graficos': [0, 15, 30, 90, 365],
    },
    {
        'nome': 'Caso 3 - Five-Spot invertido',
        'pressao_produtor': 23e6,
        'fator_permeabilidade': 0.1,
        'produtores': [(500.0, 500.0)],
        'injetores': [
            (50.0, 50.0, 25.0),
            (950.0, 50.0, 25.0),
            (50.0, 950.0, 25.0),
            (950.0, 950.0, 25.0),
        ],  # vazão de cada injetor em m³/dia
        'falha': None,
        'dias_dos_graficos': [0, 7, 15, 30, 365],
    },
]


def gerar_permeabilidade():
    gerador = np.random.RandomState(semente)
    permeabilidade_mD = gerador.lognormal(
        np.log(permeabilidade_mediana),
        desvio_log,
        (numero_celulas_y, numero_celulas_x),
    )
    return permeabilidade_mD * 1e-15


def localizar_poco(x, y, dx, dy):
    if not (0 <= x < comprimento_x and 0 <= y < comprimento_y):
        raise ValueError('O poço precisa ficar dentro do reservatório.')
    coluna = int(x / dx)
    linha = int(y / dy)
    return coluna, linha


def simular(caso, permeabilidade):
    dx = comprimento_x / numero_celulas_x
    dy = comprimento_y / numero_celulas_y
    volume_celula = dx * dy * espessura
    passo_segundos = passo_tempo * 86400
    numero_passos = round(tempo_total / passo_tempo)

    permeabilidade = permeabilidade * caso['fator_permeabilidade']
    termo_acumulacao = porosidade * compressibilidade_total / passo_segundos

    # Permeabilidade nas faces entre células: média harmônica.
    permeabilidade_x = 2 / (1 / permeabilidade[:, :-1] + 1 / permeabilidade[:, 1:])
    permeabilidade_y = 2 / (1 / permeabilidade[:-1, :] + 1 / permeabilidade[1:, :])

    # Falha selante:
    if caso['falha'] is not None:
        x_falha, y_inicio, y_fim = caso['falha']
        face_x = round(x_falha / dx)
        primeira_linha = round(y_inicio / dy)
        ultima_linha = round(y_fim / dy)
        if not np.allclose(
            [x_falha / dx, y_inicio / dy, y_fim / dy],
            [face_x, primeira_linha, ultima_linha],
        ):
            raise ValueError('A falha precisa coincidir com as faces da malha.')
        permeabilidade_x[primeira_linha:ultima_linha, face_x - 1] = 0.0

    # Índices e índice de produtividade dos poços (modelo de Peaceman):
    raio_equivalente = 0.140365 * np.hypot(dx, dy)
    if raio_poco >= raio_equivalente:
        raise ValueError('O raio do poço precisa ser menor que o raio equivalente.')

    produtores = []
    injetores = []
    celulas_ocupadas = set()

    for x, y in caso['produtores']:
        coluna, linha = localizar_poco(x, y, dx, dy)
        if (coluna, linha) in celulas_ocupadas:
            raise ValueError('Dois poços não podem ocupar a mesma célula.')
        celulas_ocupadas.add((coluna, linha))
        indice_produtividade = (
            2 * np.pi * permeabilidade[linha, coluna] * espessura
            / (viscosidade * np.log(raio_equivalente / raio_poco))
        )
        produtores.append({
            'coluna': coluna,
            'linha': linha,
            'indice': indice_produtividade,
        })

    fonte_injecao = np.zeros((numero_celulas_y, numero_celulas_x))
    for x, y, vazao_dia in caso['injetores']:
        coluna, linha = localizar_poco(x, y, dx, dy)
        if vazao_dia <= 0:
            raise ValueError('A vazão do injetor precisa ser positiva.')
        if (coluna, linha) in celulas_ocupadas:
            raise ValueError('Dois poços não podem ocupar a mesma célula.')
        celulas_ocupadas.add((coluna, linha))
        vazao_segundo = vazao_dia / 86400
        fonte_injecao[linha, coluna] += vazao_segundo / volume_celula
        indice_produtividade = (
            2 * np.pi * permeabilidade[linha, coluna] * espessura
            / (viscosidade * np.log(raio_equivalente / raio_poco))
        )
        injetores.append({
            'coluna': coluna,
            'linha': linha,
            'indice': indice_produtividade,
            'vazao': vazao_segundo,
        })

    # Montagem da matriz do sistema: A p = b.
    numero_celulas = numero_celulas_x * numero_celulas_y
    matriz = lil_matrix((numero_celulas, numero_celulas))

    for linha in range(numero_celulas_y):
        for coluna in range(numero_celulas_x):
            indice = coluna + linha * numero_celulas_x
            matriz[indice, indice] = termo_acumulacao

            if coluna > 0:
                coeficiente = permeabilidade_x[linha, coluna - 1] / (viscosidade * dx**2)
                matriz[indice, indice] += coeficiente
                matriz[indice, indice - 1] = -coeficiente
            if coluna < numero_celulas_x - 1:
                coeficiente = permeabilidade_x[linha, coluna] / (viscosidade * dx**2)
                matriz[indice, indice] += coeficiente
                matriz[indice, indice + 1] = -coeficiente
            if linha > 0:
                coeficiente = permeabilidade_y[linha - 1, coluna] / (viscosidade * dy**2)
                matriz[indice, indice] += coeficiente
                matriz[indice, indice - numero_celulas_x] = -coeficiente
            if linha < numero_celulas_y - 1:
                coeficiente = permeabilidade_y[linha, coluna] / (viscosidade * dy**2)
                matriz[indice, indice] += coeficiente
                matriz[indice, indice + numero_celulas_x] = -coeficiente

    for poco in produtores:
        indice = poco['coluna'] + poco['linha'] * numero_celulas_x
        coeficiente = poco['indice'] / volume_celula
        matriz[indice, indice] += coeficiente

    matriz = matriz.tocsr()

    # Pressão inicial e históricos de vazão e produção:
    pressao = np.full(
        (numero_passos + 1, numero_celulas_y, numero_celulas_x),
        pressao_inicial,
    )
    vazao_produtores = np.zeros((numero_passos + 1, len(produtores)))
    producao_acumulada = np.zeros(numero_passos + 1)

    for numero, poco in enumerate(produtores):
        vazao_produtores[0, numero] = (
            poco['indice'] * (pressao_inicial - caso['pressao_produtor']) * 86400
        )

    # Formulação totalmente implícita:
    for tempo in range(1, numero_passos + 1):
        vetor = termo_acumulacao * pressao[tempo - 1].ravel()
        vetor += fonte_injecao.ravel()

        for poco in produtores:
            indice = poco['coluna'] + poco['linha'] * numero_celulas_x
            vetor[indice] += (
                poco['indice'] / volume_celula * caso['pressao_produtor']
            )

        pressao[tempo] = spsolve(matriz, vetor).reshape(
            numero_celulas_y, numero_celulas_x
        )

        for numero, poco in enumerate(produtores):
            pressao_celula = pressao[tempo, poco['linha'], poco['coluna']]
            vazao_produtores[tempo, numero] = (
                poco['indice'] * (pressao_celula - caso['pressao_produtor']) * 86400
            )

        vazao_total = np.sum(vazao_produtores[tempo])
        producao_acumulada[tempo] = (
            producao_acumulada[tempo - 1] + vazao_total * passo_tempo
        )

    return {
        'caso': caso,
        'permeabilidade': permeabilidade,
        'pressao': pressao,
        'vazao_produtores': vazao_produtores,
        'producao_acumulada': producao_acumulada,
        'produtores': produtores,
        'injetores': injetores,
        'permeabilidade_x': permeabilidade_x,
        'permeabilidade_y': permeabilidade_y,
        'passo_tempo': passo_tempo,
        'dx': dx,
        'dy': dy,
    }


def desenhar_pocos(eixo, resultado):
    dx = resultado['dx']
    dy = resultado['dy']
    caso = resultado['caso']

    for numero, poco in enumerate(resultado['produtores'], start=1):
        x = (poco['coluna'] + 0.5) * dx
        y = (poco['linha'] + 0.5) * dy
        eixo.plot(x, y, 'ko', markeredgecolor='white', markersize=5)
        if len(resultado['produtores']) > 1:
            eixo.annotate(f'P{numero}', (x, y), xytext=(4, 4),
                          textcoords='offset points', fontsize=7)

    for poco in resultado['injetores']:
        x = (poco['coluna'] + 0.5) * dx
        y = (poco['linha'] + 0.5) * dy
        eixo.plot(x, y, '^', color='deepskyblue', markeredgecolor='black', markersize=6)

    if caso['falha'] is not None:
        x, y_inicio, y_fim = caso['falha']
        eixo.plot([x, x], [y_inicio, y_fim], color='crimson', linewidth=2)

    eixo.set(xlim=(0, comprimento_x), ylim=(0, comprimento_y),
             xlabel='x (m)', ylabel='y (m)')
    eixo.set_aspect('equal')


def calcular_velocidade(resultado, indice_tempo):
    pressao = resultado['pressao'][indice_tempo]
    dx = resultado['dx']
    dy = resultado['dy']

    velocidade_x = np.zeros((numero_celulas_y, numero_celulas_x + 1))
    velocidade_y = np.zeros((numero_celulas_y + 1, numero_celulas_x))
    velocidade_x[:, 1:-1] = (
        -resultado['permeabilidade_x'] / viscosidade
        * np.diff(pressao, axis=1) / dx
    )
    velocidade_y[1:-1, :] = (
        -resultado['permeabilidade_y'] / viscosidade
        * np.diff(pressao, axis=0) / dy
    )

    velocidade_x_centro = 0.5 * (velocidade_x[:, :-1] + velocidade_x[:, 1:])
    velocidade_y_centro = 0.5 * (velocidade_y[:-1, :] + velocidade_y[1:, :])
    return velocidade_x_centro * 86400, velocidade_y_centro * 86400

# Gráficos:
def graficos(resultado):
    caso = resultado['caso']
    dias = caso.get('dias_dos_graficos', dias_dos_graficos)
    permeabilidade = resultado['permeabilidade'] * 1e15
    pressao = resultado['pressao']
    dx = resultado['dx']
    dy = resultado['dy']
    indices_tempo = []

    for dia in dias:
        indice = round(dia / resultado['passo_tempo'])
        if not np.isclose(indice * resultado['passo_tempo'], dia):
            raise ValueError('Os dias dos gráficos precisam coincidir com um passo de tempo.')
        if indice >= len(pressao):
            raise ValueError('O tempo total precisa incluir os dias dos gráficos.')
        indices_tempo.append(indice)

    velocidades = [calcular_velocidade(resultado, indice) for indice in indices_tempo]
    velocidade_maxima = max(
        np.hypot(velocidade_x, velocidade_y).max()
        for velocidade_x, velocidade_y in velocidades
    )
    if velocidade_maxima == 0:
        velocidade_maxima = 1.0

    pressao_minima = min(pressao[indice].min() for indice in indices_tempo) / 1e6
    pressao_maxima = max(pressao[indice].max() for indice in indices_tempo) / 1e6
    escala_velocidade = Normalize(vmin=0, vmax=velocidade_maxima)
    extensao = [0, comprimento_x, 0, comprimento_y]

    # Campo de permeabilidade:
    figura_k, eixo_k = plt.subplots(figsize=(6, 5), constrained_layout=True)
    mapa_k = eixo_k.imshow(
        permeabilidade,
        origin='lower',
        extent=extensao,
        cmap='viridis',
        norm=LogNorm(vmin=permeabilidade.min(), vmax=permeabilidade.max()),
    )
    eixo_k.set_title(caso['nome'] + ' - Permeabilidade')
    desenhar_pocos(eixo_k, resultado)
    figura_k.colorbar(mapa_k, ax=eixo_k, label='Permeabilidade (mD)')

    # Pressão e velocidade:
    figura = plt.figure(
        figsize=(3.6 * len(dias) + 1.5, 7),
        constrained_layout=True,
    )
    figura.suptitle(caso['nome'] + ' - Pressão e velocidade')
    grade = figura.add_gridspec(
        2,
        len(dias) + 1,
        width_ratios=[1] * len(dias) + [0.06],
    )
    eixos_pressao = [figura.add_subplot(grade[0, coluna]) for coluna in range(len(dias))]
    eixos_velocidade = [figura.add_subplot(grade[1, coluna]) for coluna in range(len(dias))]

    for eixo, dia, indice in zip(eixos_pressao, dias, indices_tempo):
        mapa_pressao = eixo.imshow(
            pressao[indice] / 1e6,
            origin='lower',
            extent=extensao,
            cmap='turbo',
            vmin=pressao_minima,
            vmax=pressao_maxima,
        )
        eixo.set_title(f't = {dia} dias')
        desenhar_pocos(eixo, resultado)

    barra_pressao = figura.add_subplot(grade[0, len(dias)])
    figura.colorbar(mapa_pressao, cax=barra_pressao, label='Pressão (MPa)')

    x_centro = (np.arange(numero_celulas_x) + 0.5) * dx
    y_centro = (np.arange(numero_celulas_y) + 0.5) * dy
    X, Y = np.meshgrid(x_centro, y_centro)

    for eixo, dia, (velocidade_x, velocidade_y) in zip(
        eixos_velocidade, dias, velocidades
    ):
        modulo = np.hypot(velocidade_x, velocidade_y)
        limite = 0.01 * velocidade_maxima
        setas_x = np.zeros_like(velocidade_x)
        setas_y = np.zeros_like(velocidade_y)
        movimento = modulo > limite
        tamanho_seta = 0.7 * min(dx, dy)
        setas_x[movimento] = (
            velocidade_x[movimento] / modulo[movimento] * tamanho_seta
        )
        setas_y[movimento] = (
            velocidade_y[movimento] / modulo[movimento] * tamanho_seta
        )
        setas = eixo.quiver(
            X[::2, ::2],
            Y[::2, ::2],
            setas_x[::2, ::2],
            setas_y[::2, ::2],
            modulo[::2, ::2],
            cmap='plasma',
            norm=escala_velocidade,
            angles='xy',
            scale_units='xy',
            scale=1,
            pivot='mid',
            width=0.003,
            headwidth=3.5,
        )
        eixo.set_title(f't = {dia} dias')
        desenhar_pocos(eixo, resultado)

    barra_velocidade = figura.add_subplot(grade[1, len(dias)])
    figura.colorbar(
        setas,
        cax=barra_velocidade,
        label='Velocidade (m/dia)',
    )

    # Vazão dos produtores e produção acumulada:
    tempos = np.arange(len(pressao)) * resultado['passo_tempo']
    figura_vazao, eixo_vazao = plt.subplots(figsize=(8, 5))
    eixo_producao = eixo_vazao.twinx()

    for numero in range(resultado['vazao_produtores'].shape[1]):
        eixo_vazao.plot(
            tempos,
            resultado['vazao_produtores'][:, numero],
            label=f'Produtor {numero + 1}',
        )
    eixo_vazao.plot(
        tempos,
        resultado['vazao_produtores'].sum(axis=1),
        color='black',
        linestyle=':',
        linewidth=2,
        label='Vazão total',
    )
    eixo_producao.plot(
        tempos,
        resultado['producao_acumulada'],
        color='tab:green',
        linestyle='--',
        label='Produção acumulada',
    )
    eixo_vazao.set(
        xlabel='Tempo (dias)',
        ylabel='Vazão (m³/dia)',
        title=caso['nome'] + ' - Produção',
    )
    eixo_producao.set_ylabel('Produção acumulada (m³)')
    eixo_vazao.grid(alpha=0.3)
    linhas_1, nomes_1 = eixo_vazao.get_legend_handles_labels()
    linhas_2, nomes_2 = eixo_producao.get_legend_handles_labels()
    eixo_vazao.legend(linhas_1 + linhas_2, nomes_1 + nomes_2, loc='best')
    figura_vazao.tight_layout()


if __name__ == '__main__':
    if round(tempo_total / passo_tempo) * passo_tempo != tempo_total:
        raise ValueError('O tempo total precisa ser múltiplo do passo de tempo.')

    permeabilidade = gerar_permeabilidade()

    for caso in casos:
        print('Calculando:', caso['nome'], flush=True)
        resultado = simular(caso, permeabilidade)
        graficos(resultado)

    plt.show()
