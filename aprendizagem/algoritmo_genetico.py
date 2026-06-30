"""
Algoritmo genético com Novelty Search (versão simplificada).

Evolui os pesos de uma rede feedforward (o "genoma") para resolver um ambiente.
Combina a fitness (recompensa total) com uma medida de novidade comportamental,
para evitar que a população convirja cedo demais para um comportamento pobre.

Mantemos o algoritmo deliberadamente simples: torneio binário, dois operadores
de crossover, mutação gaussiana e um pouco de elitismo. O descritor de
comportamento é compacto (posição final, cobertura e distância final).
"""

import numpy as np

from core.agente import Agente
from core.simulador import Simulador
from core.metricas import recompensa_total
from agentes.genetico import AgenteGenetico
from aprendizagem.rede import RedePolitica


# ----------------------------------------------------------------------
# Avaliação de um genoma
# ----------------------------------------------------------------------
def _descritor(ep, ambiente):
    """Vetor que resume o comportamento de um episódio (para a novelty)."""
    traj = ep["trajetoria"]
    fim = traj[-1]
    n = max(ambiente.n_linhas, ambiente.n_colunas)
    visitadas = len(set(traj))
    return np.array([
        fim[0] / n,
        fim[1] / n,
        visitadas / (n * n),
        ep["passos"] / ambiente.max_passos,
    ], dtype=np.float64)


def avalia_genoma(genoma, env_factory, n_entradas, n_escondidas, n_episodios=1):
    ambiente = env_factory()
    agente = AgenteGenetico("ga", n_entradas, n_escondidas, genoma=genoma)
    sim = Simulador(ambiente)

    fit_total = 0.0
    descritor = None
    for _ in range(n_episodios):
        ep = sim.corre_episodio(agente, treino=False)
        fit_total += recompensa_total(ep)
        descritor = _descritor(ep, ambiente)   # guardamos o do último episódio

    return fit_total / n_episodios, descritor


# ----------------------------------------------------------------------
# Operadores genéticos
# ----------------------------------------------------------------------
def crossover(p1, p2, rng):
    if rng.random() < 0.5:
        # uniforme
        mascara = rng.random(len(p1)) < 0.5
        return np.where(mascara, p1, p2)
    # um ponto
    ponto = rng.integers(1, len(p1))
    return np.concatenate([p1[:ponto], p2[ponto:]])


def muta(genoma, rng, taxa=0.1, sigma=0.3):
    novo = genoma.copy()
    mascara = rng.random(len(genoma)) < taxa
    ruido = rng.normal(0, sigma, size=len(genoma))
    novo[mascara] += ruido[mascara]
    return novo


def novelty(descritores, k=8):
    """Distância média aos k vizinhos mais próximos."""
    nov = np.zeros(len(descritores))
    for i, d in enumerate(descritores):
        dists = sorted(np.linalg.norm(d - outro) for j, outro in enumerate(descritores) if j != i)
        kk = min(k, len(dists))
        nov[i] = np.mean(dists[:kk]) if kk > 0 else 0.0
    return nov


def _normaliza(v):
    v = np.asarray(v, dtype=np.float64)
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-9:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


# ----------------------------------------------------------------------
# Ciclo evolutivo
# ----------------------------------------------------------------------
def treina(env_factory, n_entradas, n_escondidas=16, pop_size=60, geracoes=200,
           episodios=1, taxa_mut=0.1, sigma=0.3, elite=3, alpha=0.5, seed=42,
           verboso=True):
    rng = np.random.default_rng(seed)

    # tamanho do genoma
    modelo = RedePolitica(n_entradas, n_escondidas, Agente.N_ACOES, rng=rng)
    dim = modelo.n_pesos

    populacao = [rng.normal(0, 0.5, size=dim) for _ in range(pop_size)]
    arquivo = []                      # descritores históricos (novelty)
    historico = []
    melhor_global = (None, -np.inf)   # (genoma, fitness)

    for g in range(1, geracoes + 1):
        # alpha decresce: começa a explorar, acaba a otimizar
        a = alpha
        if g > geracoes * 0.7:
            a = max(0.2, alpha - 0.3)

        fits, descritores = [], []
        for genoma in populacao:
            f, d = avalia_genoma(genoma, env_factory, n_entradas, n_escondidas, episodios)
            fits.append(f)
            descritores.append(d)

        fits = np.array(fits)
        descr_para_novelty = descritores + arquivo
        nov = novelty(descr_para_novelty)[:len(populacao)]

        combinado = a * _normaliza(nov) + (1 - a) * _normaliza(fits)

        # registar histórico
        historico.append({
            "geracao": g,
            "fitness_media": float(fits.mean()),
            "fitness_max": float(fits.max()),
            "novelty_media": float(nov.mean()),
            "novelty_max": float(nov.max()),
        })

        # guardar melhor de sempre (por fitness)
        idx_melhor = int(np.argmax(fits))
        if fits[idx_melhor] > melhor_global[1]:
            melhor_global = (populacao[idx_melhor].copy(), float(fits[idx_melhor]))

        if verboso and (g % 10 == 0 or g == 1):
            print(f"  ger {g:4d}/{geracoes} | fit med {fits.mean():7.2f} "
                  f"max {fits.max():7.2f} | nov med {nov.mean():.3f}")

        # elitismo
        ordem = np.argsort(combinado)[::-1]
        nova_pop = [populacao[i].copy() for i in ordem[:elite]]

        # arquivo de novelty (guarda alguns descritores ao acaso)
        for d in descritores:
            if rng.random() < 0.1:
                arquivo.append(d)

        # descendência
        while len(nova_pop) < pop_size:
            i, j = rng.integers(0, pop_size, 2)
            pai = populacao[i] if combinado[i] > combinado[j] else populacao[j]
            i, j = rng.integers(0, pop_size, 2)
            mae = populacao[i] if combinado[i] > combinado[j] else populacao[j]
            filho = crossover(pai, mae, rng)
            filho = muta(filho, rng, taxa_mut, sigma)
            nova_pop.append(filho)

        populacao = nova_pop

    return historico, melhor_global[0], melhor_global[1]
