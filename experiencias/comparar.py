"""
Modo de Teste: compara os três tipos de agente (Fixo, Q-learning e Genético)
nos dois ambientes, com política fixa/pré-treinada.

  - Labirinto: avaliação no labirinto fixo de cada dificuldade (o mesmo em que
    o Q-learning foi treinado). Mede passos e taxa de sucesso.
  - Foraging: avaliação em vários mundos gerados com seeds diferentes, para
    medir o desempenho médio (pontos depositados) e a taxa de "limpeza".

Gera os gráficos de barras de comparação e um resumo em JSON para o relatório.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.simulador import Simulador
from core.metricas import resume
from ambientes.labirinto import AmbienteLabirinto
from ambientes.foraging import AmbienteForaging
from agentes.fixo_labirinto import AgenteFixoLabirinto
from agentes.fixo_foraging import AgenteFixoForaging
from agentes.qlearning import AgenteQLearning
from agentes.genetico import AgenteGenetico
from aprendizagem.discretizadores import maze_discretizador, foraging_discretizador
from configs import labirinto as cfg_lab
from configs import foraging as cfg_for
from experiencias.comum import caminho

CORES = {"Fixo": "#888888", "Genético": "#4169e1", "Q-learning": "#2b8a3e"}


def avalia(ambiente_factory, agente, n_episodios):
    sim = Simulador(ambiente_factory())
    episodios = []
    for _ in range(n_episodios):
        sim.ambiente = ambiente_factory()   # mundo novo a cada episódio
        episodios.append(sim.corre_episodio(agente))
    return episodios


# ----------------------------------------------------------------------
def compara_labirinto():
    print("\n##### TESTE — LABIRINTO #####")
    resultados = {}

    for dif in (0, 1, 2):
        p = cfg_lab.NIVEIS[dif]
        fab = lambda p=p: AmbienteLabirinto(lado=p["lado"], seed=p["seed"], max_passos=p["max_passos"])

        agentes = {
            "Fixo": AgenteFixoLabirinto("fixo"),
            "Genético": AgenteGenetico.carrega(caminho("ga_labirinto.npy"), n_entradas=8),
            "Q-learning": _ql_lab(dif),
        }

        resultados[dif] = {}
        for nome, ag in agentes.items():
            eps = avalia(fab, ag, n_episodios=1)   # labirinto fixo -> determinístico
            r = resume(eps)
            resultados[dif][nome] = r
            print(f"  dif {dif} | {nome:11s}: sucesso {r['taxa_sucesso']:5.1f}% "
                  f"passos {r['passos_medios']:.0f}")

    _grafico_labirinto(resultados)
    return resultados


def _ql_lab(dif):
    ag = AgenteQLearning("ql", maze_discretizador(cfg_lab.NIVEIS[dif]["lado"]), epsilon=0.0)
    ag.carrega_tabela(caminho(f"ql_labirinto_dif{dif}.pkl"))
    return ag


def _grafico_labirinto(resultados):
    difs = [0, 1, 2]
    nomes = ["Fixo", "Genético", "Q-learning"]
    x = np.arange(len(difs))
    largura = 0.25

    # taxa de sucesso
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, nome in enumerate(nomes):
        vals = [resultados[d][nome]["taxa_sucesso"] for d in difs]
        ax.bar(x + (i - 1) * largura, vals, largura, label=nome, color=CORES[nome])
    ax.set_xticks(x); ax.set_xticklabels([f"dif {d}" for d in difs])
    ax.set_ylabel("Taxa de sucesso (%)"); ax.set_ylim(0, 105)
    ax.set_title("Labirinto — Taxa de sucesso por dificuldade")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(caminho("comparacao_labirinto_sucesso.png"), dpi=130)
    plt.close(fig)

    # passos médios
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, nome in enumerate(nomes):
        vals = [resultados[d][nome]["passos_medios"] for d in difs]
        ax.bar(x + (i - 1) * largura, vals, largura, label=nome, color=CORES[nome])
    ax.set_xticks(x); ax.set_xticklabels([f"dif {d}" for d in difs])
    ax.set_ylabel("Passos médios até terminar")
    ax.set_title("Labirinto — Passos médios por dificuldade")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(caminho("comparacao_labirinto_passos.png"), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
def compara_labirinto_generalizacao(n_mundos=30):
    """Avaliação em labirintos aleatórios NÃO vistos no treino, para ver a
    capacidade de generalização de cada agente."""
    print("\n##### TESTE — LABIRINTO (generalização, mapas não vistos) #####")
    resultados = {}

    for dif in (0, 1, 2):
        p = cfg_lab.NIVEIS[dif]
        fab = lambda p=p: AmbienteLabirinto(lado=p["lado"], max_passos=p["max_passos"], aleatorio=True)

        agentes = {
            "Fixo": AgenteFixoLabirinto("fixo"),
            "Genético": AgenteGenetico.carrega(caminho("ga_labirinto.npy"), n_entradas=8),
            "Q-learning": _ql_lab(dif),
        }

        resultados[dif] = {}
        for nome, ag in agentes.items():
            eps = avalia(fab, ag, n_episodios=n_mundos)
            r = resume(eps)
            resultados[dif][nome] = r
            print(f"  dif {dif} | {nome:11s}: sucesso {r['taxa_sucesso']:5.1f}% "
                  f"em {n_mundos} mapas novos")

    difs = [0, 1, 2]
    nomes = ["Fixo", "Genético", "Q-learning"]
    x = np.arange(len(difs))
    largura = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, nome in enumerate(nomes):
        vals = [resultados[d][nome]["taxa_sucesso"] for d in difs]
        ax.bar(x + (i - 1) * largura, vals, largura, label=nome, color=CORES[nome])
    ax.set_xticks(x); ax.set_xticklabels([f"dif {d}" for d in difs])
    ax.set_ylabel("Taxa de sucesso (%)"); ax.set_ylim(0, 105)
    ax.set_title("Labirinto — Generalização a mapas não vistos")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(caminho("comparacao_labirinto_generalizacao.png"), dpi=130)
    plt.close(fig)
    return resultados


# ----------------------------------------------------------------------
def compara_foraging():
    print("\n##### TESTE — FORAGING #####")
    seeds = list(range(1000, 1015))   # 15 mundos por dificuldade
    resultados = {}

    for dif in (0, 1):
        p = cfg_for.NIVEIS[dif]

        agentes = {
            "Fixo": AgenteFixoForaging("fixo"),
            "Genético": AgenteGenetico.carrega(caminho("ga_foraging.npy"), n_entradas=7),
            "Q-learning": _ql_for(),
        }

        resultados[dif] = {}
        for nome, ag in agentes.items():
            eps = []
            sim = Simulador(AmbienteForaging(lado=p["lado"]))
            for s in seeds:
                sim.ambiente = AmbienteForaging(lado=p["lado"], seed=s, n_recursos=p["n_recursos"],
                                                prob_obstaculo=p["prob_obstaculo"], max_passos=p["max_passos"])
                eps.append(sim.corre_episodio(ag))
            pontos = [e["score"] for e in eps]
            r = resume(eps)
            r["pontos_medios"] = float(np.mean(pontos))
            r["pontos_std"] = float(np.std(pontos))
            resultados[dif][nome] = r
            print(f"  dif {dif} | {nome:11s}: pontos {r['pontos_medios']:.2f} "
                  f"(+/-{r['pontos_std']:.2f}) | limpeza {r['taxa_sucesso']:.0f}%")

    _grafico_foraging(resultados)
    return resultados


def _ql_for():
    ag = AgenteQLearning("qlf", foraging_discretizador(), epsilon=0.0)
    ag.carrega_tabela(caminho("ql_foraging.pkl"))
    return ag


def _grafico_foraging(resultados):
    difs = [0, 1]
    nomes = ["Fixo", "Genético", "Q-learning"]
    x = np.arange(len(difs))
    largura = 0.25

    # pontos depositados
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, nome in enumerate(nomes):
        vals = [resultados[d][nome]["pontos_medios"] for d in difs]
        erros = [resultados[d][nome]["pontos_std"] for d in difs]
        ax.bar(x + (i - 1) * largura, vals, largura, yerr=erros, capsize=4,
               label=nome, color=CORES[nome])
    ax.set_xticks(x); ax.set_xticklabels([f"dif {d}" for d in difs])
    ax.set_ylabel("Pontos depositados (média de 15 mundos)")
    ax.set_title("Foraging — Pontos depositados")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(caminho("comparacao_foraging_pontos.png"), dpi=130)
    plt.close(fig)

    # taxa de limpeza
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, nome in enumerate(nomes):
        vals = [resultados[d][nome]["taxa_sucesso"] for d in difs]
        ax.bar(x + (i - 1) * largura, vals, largura, label=nome, color=CORES[nome])
    ax.set_xticks(x); ax.set_xticklabels([f"dif {d}" for d in difs])
    ax.set_ylabel("Taxa de limpeza total (%)"); ax.set_ylim(0, 105)
    ax.set_title("Foraging — Episódios com mapa totalmente limpo")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(caminho("comparacao_foraging_sucesso.png"), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
def _serializa(d):
    return {str(k): v for k, v in d.items()}


if __name__ == "__main__":
    lab = compara_labirinto()
    lab_gen = compara_labirinto_generalizacao()
    fora = compara_foraging()
    resumo = {"labirinto": {str(k): v for k, v in lab.items()},
              "labirinto_generalizacao": {str(k): v for k, v in lab_gen.items()},
              "foraging": {str(k): v for k, v in fora.items()}}
    with open(caminho("resumo_testes.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    print("\nComparações guardadas em resultados/.")
