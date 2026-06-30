"""
Treino dos agentes que aprendem no ambiente Labirinto.

  - Q-learning: treinado em cada labirinto fixo (memoriza o mapa). Guarda uma
    tabela Q por dificuldade.
  - Genético: treinado em labirintos aleatórios pequenos, para obter uma
    política reativa que generalize. Guarda um único genoma campeão.

Gera as curvas de aprendizagem e guarda os modelos em resultados/.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ambientes.labirinto import AmbienteLabirinto
from agentes.qlearning import AgenteQLearning
from aprendizagem.discretizadores import maze_discretizador
from aprendizagem import qlearning_treino, algoritmo_genetico
from configs import labirinto as cfg
from experiencias.comum import caminho, media_movel

# nº de episódios de Q-learning por dificuldade (modesto, de propósito)
EPISODIOS_QL = {0: 2000, 1: 3500, 2: 4500}


def treina_qlearning():
    print("\n##### Q-LEARNING — LABIRINTO #####")
    plt.figure(figsize=(9, 5))

    for dif in (0, 1, 2):
        p = cfg.NIVEIS[dif]
        print(f"\n-- dificuldade {dif} ({p['lado']}x{p['lado']}) --")
        amb = AmbienteLabirinto(lado=p["lado"], seed=p["seed"], max_passos=p["max_passos"])
        agente = AgenteQLearning(f"ql{dif}", maze_discretizador(p["lado"]),
                                 taxa=0.2, desconto=0.95, epsilon=1.0)

        hist = qlearning_treino.treina(amb, agente, n_episodios=EPISODIOS_QL[dif],
                                       epsilon_min=0.05, decaimento=0.9985)

        agente.guarda(caminho(f"ql_labirinto_dif{dif}.pkl"))
        with open(caminho(f"hist_ql_labirinto_dif{dif}.json"), "w", encoding="utf-8") as f:
            json.dump(hist, f)

        rec = [h["recompensa"] for h in hist]
        suav = media_movel(rec, 100)
        plt.plot(range(len(suav)), suav, label=f"dificuldade {dif}")

    plt.xlabel("Episódio")
    plt.ylabel("Recompensa (média móvel)")
    plt.title("Curva de aprendizagem — Q-learning (Labirinto)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho("curva_ql_labirinto.png"), dpi=130)
    plt.close()


def treina_genetico():
    print("\n##### GENÉTICO — LABIRINTO #####")

    def fabrica():
        # labirintos pequenos aleatórios (generalização)
        lado = int(np.random.choice([11, 13]))
        return AmbienteLabirinto(lado=lado, max_passos=150, aleatorio=True)

    hist, melhor, fit = algoritmo_genetico.treina(
        fabrica, n_entradas=8, n_escondidas=16, pop_size=60, geracoes=150,
        episodios=1, taxa_mut=0.12, sigma=0.3, elite=3, alpha=0.5, seed=10)

    np.save(caminho("ga_labirinto.npy"), melhor)
    with open(caminho("hist_ga_labirinto.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f)
    print(f"  melhor fitness do campeão: {fit:.2f}")

    gers = [h["geracao"] for h in hist]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(gers, [h["fitness_media"] for h in hist], label="média")
    ax1.plot(gers, [h["fitness_max"] for h in hist], label="máxima")
    ax1.set_title("Fitness por geração"); ax1.set_xlabel("Geração"); ax1.set_ylabel("Fitness")
    ax1.grid(True, alpha=0.3); ax1.legend()
    ax2.plot(gers, [h["novelty_media"] for h in hist], color="tab:orange", label="média")
    ax2.plot(gers, [h["novelty_max"] for h in hist], color="tab:red", label="máxima")
    ax2.set_title("Novelty por geração"); ax2.set_xlabel("Geração"); ax2.set_ylabel("Novelty")
    ax2.grid(True, alpha=0.3); ax2.legend()
    fig.suptitle("Curva de aprendizagem — Genético (Labirinto)")
    fig.tight_layout()
    fig.savefig(caminho("curva_ga_labirinto.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    treina_qlearning()
    treina_genetico()
    print("\nConcluído. Modelos e curvas em resultados/.")
