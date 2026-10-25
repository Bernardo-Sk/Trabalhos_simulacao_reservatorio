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

# Condições de contorno para cada lado do domínio:
condicoes_contorno = {
    'oeste': {'tipo': 'fluxo', 'valor': 0.0},
    'leste': {'tipo': 'fluxo', 'valor': 0.0},
    'sul': {'tipo': 'fluxo', 'valor': 0.0},
    'norte': {'tipo': 'fluxo', 'valor': 0.0},
}

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
    # Conversão aproximada de mD para m²
    return permeabilidade_mD * 1e-15


def localizar_poco(x, y, dx, dy):
    if not (0 <= x < comprimento_x and 0 <= y < comprimento_y):
        raise ValueError('O poço precisa ficar dentro do reservatório.')
    coluna = int(x / dx)
    linha = int(y / dy)
    return coluna, linha


def transmissibilidade(permeabilidade_face, area, distancia):
    return permeabilidade_face * area / (viscosidade * distancia)


def validar_contorno(contorno):
    lados = ('oeste', 'leste', 'sul', 'norte')
    for lado in lados:
        if lado not in contorno:
            raise ValueError(f'Falta definir a condição de contorno: {lado}.')
        condicao = contorno[lado]
        if condicao['tipo'] not in ('pressao', 'fluxo'):
            raise ValueError('A condição de contorno deve ser pressão ou fluxo.')


def simular(caso, permeabilidade_base):
    dx = comprimento_x / numero_celulas_x
    dy = comprimento_y / numero_celulas_y
    volume_celula = dx * dy * espessura
    area_face_x = dy * espessura
    area_face_y = dx * espessura
    passo_segundos = passo_tempo * 86400.0
    numero_passos = round(tempo_total / passo_tempo)

    permeabilidade = permeabilidade_base * caso['fator_permeabilidade']
    contorno = caso.get('contorno', condicoes_contorno)
    validar_contorno(contorno)

    # Permeabilidade nas faces internas:
    permeabilidade_x = 2.0 / (
        1.0 / permeabilidade[:, :-1] + 1.0 / permeabilidade[:, 1:]
    )
    permeabilidade_y = 2.0 / (
        1.0 / permeabilidade[:-1, :] + 1.0 / permeabilidade[1:, :]
    )

    # Falha:
    mascara_falha = np.zeros((numero_celulas_y, numero_celulas_x - 1), dtype=bool)
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
        if not (1 <= face_x < numero_celulas_x):
            raise ValueError('A falha precisa estar em uma face interna da malha.')
        mascara_falha[primeira_linha:ultima_linha, face_x - 1] = True

    # Poços produtores com pressão de fundo prescrita (modelo de Peaceman):
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
        indice_poco = (
            2.0 * np.pi * permeabilidade[linha, coluna] * espessura
            / (viscosidade * np.log(raio_equivalente / raio_poco))
        )
        produtores.append({
            'coluna': coluna,
            'linha': linha,
            'indice': indice_poco,
        })

    fonte_injecao = np.zeros((numero_celulas_y, numero_celulas_x))
    for x, y, vazao_dia in caso['injetores']:
        coluna, linha = localizar_poco(x, y, dx, dy)
        if vazao_dia <= 0:
            raise ValueError('A vazão do injetor precisa ser positiva.')
        if (coluna, linha) in celulas_ocupadas:
            raise ValueError('Dois poços não podem ocupar a mesma célula.')
        celulas_ocupadas.add((coluna, linha))
        vazao_segundo = vazao_dia / 86400.0
        fonte_injecao[linha, coluna] += vazao_segundo
        injetores.append({
            'coluna': coluna,
            'linha': linha,
            'vazao': vazao_segundo,
        })

    # Volumes finitos:
    numero_celulas = numero_celulas_x * numero_celulas_y
    matriz = lil_matrix((numero_celulas, numero_celulas))
    armazenamento = (
        porosidade * compressibilidade_total * volume_celula / passo_segundos
    )

    for linha in range(numero_celulas_y):
        for coluna in range(numero_celulas_x):
            indice = coluna + linha * numero_celulas_x
            matriz[indice, indice] = armazenamento

            # Faces internas leste/oeste:
            if coluna > 0 and not mascara_falha[linha, coluna - 1]:
                coeficiente = transmissibilidade(
                    permeabilidade_x[linha, coluna - 1], area_face_x, dx
                )
                matriz[indice, indice] += coeficiente
                matriz[indice, indice - 1] -= coeficiente
            if coluna < numero_celulas_x - 1 and not mascara_falha[linha, coluna]:
                coeficiente = transmissibilidade(
                    permeabilidade_x[linha, coluna], area_face_x, dx
                )
                matriz[indice, indice] += coeficiente
                matriz[indice, indice + 1] -= coeficiente

            # Faces internas norte/sul:
            if linha > 0:
                coeficiente = transmissibilidade(
                    permeabilidade_y[linha - 1, coluna], area_face_y, dy
                )
                matriz[indice, indice] += coeficiente
                matriz[indice, indice - numero_celulas_x] -= coeficiente
            if linha < numero_celulas_y - 1:
                coeficiente = transmissibilidade(
                    permeabilidade_y[linha, coluna], area_face_y, dy
                )
                matriz[indice, indice] += coeficiente
                matriz[indice, indice + numero_celulas_x] -= coeficiente

            # Condições nas faces externas:
            if coluna == 0:
                condicao = contorno['oeste']
                if condicao['tipo'] == 'pressao':
                    coeficiente = transmissibilidade(
                        permeabilidade[linha, coluna], area_face_x, dx / 2.0
                    )
                    matriz[indice, indice] += coeficiente
            if coluna == numero_celulas_x - 1:
                condicao = contorno['leste']
                if condicao['tipo'] == 'pressao':
                    coeficiente = transmissibilidade(
                        permeabilidade[linha, coluna], area_face_x, dx / 2.0
                    )
                    matriz[indice, indice] += coeficiente
            if linha == 0:
                condicao = contorno['sul']
                if condicao['tipo'] == 'pressao':
                    coeficiente = transmissibilidade(
                        permeabilidade[linha, coluna], area_face_y, dy / 2.0
                    )
                    matriz[indice, indice] += coeficiente
            if linha == numero_celulas_y - 1:
                condicao = contorno['norte']
                if condicao['tipo'] == 'pressao':
                    coeficiente = transmissibilidade(
                        permeabilidade[linha, coluna], area_face_y, dy / 2.0
                    )
                    matriz[indice, indice] += coeficiente

    for poco in produtores:
        indice = poco['coluna'] + poco['linha'] * numero_celulas_x
        matriz[indice, indice] += poco['indice']

    matriz = matriz.tocsr()

    pressao = np.full(
        (numero_passos + 1, numero_celulas_y, numero_celulas_x),
        pressao_inicial,
    )
    vazao_produtores = np.zeros((numero_passos + 1, len(produtores)))
    vazao_injetores = np.zeros((numero_passos + 1, len(injetores)))
    producao_acumulada = np.zeros(numero_passos + 1)

    for numero, poco in enumerate(produtores):
        p_celula = pressao_inicial
        vazao_produtores[0, numero] = (
            poco['indice'] * (p_celula - caso['pressao_produtor']) * 86400.0
        )
    for numero, poco in enumerate(injetores):
        vazao_injetores[:, numero] = -poco['vazao'] * 86400.0

    # Formulação totalmente implícita:
    for tempo in range(1, numero_passos + 1):
        vetor = armazenamento * pressao[tempo - 1].ravel()
        vetor += fonte_injecao.ravel()

        # Vazões prescritas nas fronteiras:
        for linha in range(numero_celulas_y):
            for coluna, lado in ((0, 'oeste'), (numero_celulas_x - 1, 'leste')):
                condicao = contorno[lado]
                indice = coluna + linha * numero_celulas_x
                if condicao['tipo'] == 'fluxo':
                    vetor[indice] += condicao['valor']
                else:
                    coeficiente = transmissibilidade(
                        permeabilidade[linha, coluna], area_face_x, dx / 2.0
                    )
                    vetor[indice] += coeficiente * condicao['valor']
        for coluna in range(numero_celulas_x):
            for linha, lado in ((0, 'sul'), (numero_celulas_y - 1, 'norte')):
                condicao = contorno[lado]
                indice = coluna + linha * numero_celulas_x
                if condicao['tipo'] == 'fluxo':
                    vetor[indice] += condicao['valor']
                else:
                    coeficiente = transmissibilidade(
                        permeabilidade[linha, coluna], area_face_y, dy / 2.0
                    )
                    vetor[indice] += coeficiente * condicao['valor']

        for poco in produtores:
            indice = poco['coluna'] + poco['linha'] * numero_celulas_x
            vetor[indice] += poco['indice'] * caso['pressao_produtor']

        pressao[tempo] = spsolve(matriz, vetor).reshape(
            numero_celulas_y, numero_celulas_x
        )

        for numero, poco in enumerate(produtores):
            p_celula = pressao[tempo, poco['linha'], poco['coluna']]
            vazao_produtores[tempo, numero] = (
                poco['indice'] * (p_celula - caso['pressao_produtor']) * 86400.0
            )

        vazao_total = vazao_produtores[tempo].sum()
        producao_acumulada[tempo] = (
            producao_acumulada[tempo - 1] + vazao_total * passo_tempo
        )

    return {
        'caso': caso,
        'contorno': contorno,
        'permeabilidade': permeabilidade,
        'permeabilidade_x': permeabilidade_x,
        'permeabilidade_y': permeabilidade_y,
        'mascara_falha': mascara_falha,
        'pressao': pressao,
        'vazao_produtores': vazao_produtores,
        'vazao_injetores': vazao_injetores,
        'producao_acumulada': producao_acumulada,
        'produtores': produtores,
        'injetores': injetores,
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
    permeabilidade = resultado['permeabilidade']
    vx_face = np.zeros((numero_celulas_y, numero_celulas_x + 1))
    vy_face = np.zeros((numero_celulas_y + 1, numero_celulas_x))

    # Velocidades de Darcy nas faces internas:
    vx_face[:, 1:-1] = (
        -resultado['permeabilidade_x'] / viscosidade
        * np.diff(pressao, axis=1) / dx
    )
    vy_face[1:-1, :] = (
        -resultado['permeabilidade_y'] / viscosidade
        * np.diff(pressao, axis=0) / dy
    )
    vx_face[:, 1:-1] = np.where(
        resultado['mascara_falha'], 0.0, vx_face[:, 1:-1]
    )

    # Faces externas com pressão prescrita ou fluxo prescrito:
    contorno = resultado['contorno']
    for linha in range(numero_celulas_y):
        oeste = contorno['oeste']
        leste = contorno['leste']
        if oeste['tipo'] == 'pressao':
            vx_face[linha, 0] = (
                -permeabilidade[linha, 0] / viscosidade
                * (pressao[linha, 0] - oeste['valor']) / (dx / 2.0)
            )
        else:
            vx_face[linha, 0] = oeste['valor'] / (dy * espessura)
        if leste['tipo'] == 'pressao':
            vx_face[linha, -1] = (
                -permeabilidade[linha, -1] / viscosidade
                * (leste['valor'] - pressao[linha, -1]) / (dx / 2.0)
            )
        else:
            vx_face[linha, -1] = -leste['valor'] / (dy * espessura)

    for coluna in range(numero_celulas_x):
        sul = contorno['sul']
        norte = contorno['norte']
        if sul['tipo'] == 'pressao':
            vy_face[0, coluna] = (
                -permeabilidade[0, coluna] / viscosidade
                * (pressao[0, coluna] - sul['valor']) / (dy / 2.0)
            )
        else:
            vy_face[0, coluna] = sul['valor'] / (dx * espessura)
        if norte['tipo'] == 'pressao':
            vy_face[-1, coluna] = (
                -permeabilidade[-1, coluna] / viscosidade
                * (norte['valor'] - pressao[-1, coluna]) / (dy / 2.0)
            )
        else:
            vy_face[-1, coluna] = -norte['valor'] / (dx * espessura)

    vx_centro = 0.5 * (vx_face[:, :-1] + vx_face[:, 1:])
    vy_centro = 0.5 * (vy_face[:-1, :] + vy_face[1:, :])
    return vx_centro * 86400.0, vy_centro * 86400.0

# Gráficos:
def graficos(resultado):
    caso = resultado['caso']
    dias = caso['dias_dos_graficos']
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

    permeabilidade_mD = resultado['permeabilidade'] * 1e15
    extensao = [0, comprimento_x, 0, comprimento_y]

    # Campo de permeabilidade:
    figura_k, eixo_k = plt.subplots(figsize=(6, 5), constrained_layout=True)
    mapa_k = eixo_k.imshow(
        permeabilidade_mD,
        origin='lower',
        extent=extensao,
        cmap='viridis',
        norm=LogNorm(vmin=permeabilidade_mD.min(), vmax=permeabilidade_mD.max()),
    )
    eixo_k.set_title(caso['nome'] + ' - Permeabilidade')
    desenhar_pocos(eixo_k, resultado)
    figura_k.colorbar(mapa_k, ax=eixo_k, label='Permeabilidade (mD)')

    pressao_minima = min(pressao[indice].min() for indice in indices_tempo) / 1e6
    pressao_maxima = max(pressao[indice].max() for indice in indices_tempo) / 1e6
    velocidades = [calcular_velocidade(resultado, i) for i in indices_tempo]
    velocidade_maxima = max(
        np.hypot(vx, vy).max() for vx, vy in velocidades
    )
    if velocidade_maxima == 0:
        velocidade_maxima = 1.0
    escala_velocidade = Normalize(vmin=0.0, vmax=velocidade_maxima)

    figura = plt.figure(figsize=(3.6 * len(dias) + 1.5, 7), constrained_layout=True)
    figura.suptitle(caso['nome'] + ' - Pressão e velocidade')
    grade = figura.add_gridspec(
        2, len(dias) + 1, width_ratios=[1] * len(dias) + [0.06]
    )
    eixos_pressao = [figura.add_subplot(grade[0, j]) for j in range(len(dias))]
    eixos_velocidade = [figura.add_subplot(grade[1, j]) for j in range(len(dias))]

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
    figura.colorbar(
        mapa_pressao,
        cax=figura.add_subplot(grade[0, len(dias)]),
        label='Pressão (MPa)',
    )

    x_centro = (np.arange(numero_celulas_x) + 0.5) * dx
    y_centro = (np.arange(numero_celulas_y) + 0.5) * dy
    X, Y = np.meshgrid(x_centro, y_centro)
    for eixo, dia, (vx, vy) in zip(eixos_velocidade, dias, velocidades):
        modulo = np.hypot(vx, vy)
        limite = 0.01 * velocidade_maxima
        setas_x = np.zeros_like(vx)
        setas_y = np.zeros_like(vy)
        movimento = modulo > limite
        tamanho_seta = 0.7 * min(dx, dy)
        setas_x[movimento] = vx[movimento] / modulo[movimento] * tamanho_seta
        setas_y[movimento] = vy[movimento] / modulo[movimento] * tamanho_seta
        setas = eixo.quiver(
            X[::2, ::2], Y[::2, ::2],
            setas_x[::2, ::2], setas_y[::2, ::2], modulo[::2, ::2],
            cmap='plasma', norm=escala_velocidade, angles='xy',
            scale_units='xy', scale=1, pivot='mid', width=0.003,
            headwidth=3.5,
        )
        eixo.set_title(f't = {dia} dias')
        desenhar_pocos(eixo, resultado)
    figura.colorbar(
        setas,
        cax=figura.add_subplot(grade[1, len(dias)]),
        label='Velocidade (m/dia)',
    )

    # Vazões dos poços e produção acumulada no mesmo gráfico:
    tempos = np.arange(len(pressao)) * resultado['passo_tempo']
    figura_producao, eixo_vazao = plt.subplots(figsize=(8, 5))
    eixo_acumulada = eixo_vazao.twinx()
    for numero in range(resultado['vazao_produtores'].shape[1]):
        eixo_vazao.plot(
            tempos, resultado['vazao_produtores'][:, numero],
            label=f'Produtor {numero + 1}',
        )

    # A injeção aparece negativa para indicar entrada no reservatório:
    if resultado['vazao_injetores'].shape[1] > 0:
        vazao_por_injetor = resultado['vazao_injetores'][:, 0]
        eixo_vazao.plot(
            tempos, vazao_por_injetor,
            color='tab:orange', linestyle='--',
            label=(
                f'Injeção por poço ({resultado["vazao_injetores"].shape[1]} poços; '
                f'{abs(vazao_por_injetor[0]):g} m³/dia cada)'
            ),
        )
        eixo_vazao.plot(
            tempos, resultado['vazao_injetores'].sum(axis=1),
            color='black', linestyle=':', linewidth=2,
            label='Injeção total',
        )

    if resultado['vazao_produtores'].shape[1] > 1:
        eixo_vazao.plot(
            tempos, resultado['vazao_produtores'].sum(axis=1),
            color='black', linestyle=':', linewidth=2,
            label='Produção total',
        )

    eixo_acumulada.plot(
        tempos, resultado['producao_acumulada'],
        color='tab:green', linestyle='--', label='Produção acumulada',
    )
    eixo_vazao.set(
        xlabel='Tempo (dias)',
        ylabel='Vazão (m³/dia)',
        title=caso['nome'] + ' - Produção',
    )
    eixo_acumulada.set_ylabel('Produção acumulada (m³)')
    eixo_vazao.grid(alpha=0.3)
    linhas_vazao, nomes_vazao = eixo_vazao.get_legend_handles_labels()
    linhas_acumulada, nomes_acumulada = eixo_acumulada.get_legend_handles_labels()
    eixo_vazao.legend(
        linhas_vazao + linhas_acumulada,
        nomes_vazao + nomes_acumulada,
        loc='best',
    )
    figura_producao.tight_layout()

if __name__ == '__main__':
    if passo_tempo <= 0 or tempo_total <= 0:
        raise ValueError('Os tempos precisam ser positivos.')
    if not np.isclose(round(tempo_total / passo_tempo) * passo_tempo, tempo_total):
        raise ValueError('O tempo total precisa ser múltiplo do passo de tempo.')

    permeabilidade = gerar_permeabilidade()
    for caso in casos:
        print('Calculando:', caso['nome'], flush=True)
        resultado = simular(caso, permeabilidade)
        graficos(resultado)
    plt.show()
