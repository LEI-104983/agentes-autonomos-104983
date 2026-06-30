"""
Treino dos agentes que aprendem no ambiente de Recoleção (Foraging).

  - Q-learning: usa um estado reativo (paredes à volta, se transporta recurso e
    direção do alvo), por isso aprende uma política geral de recolha. Treinado
    em mundos aleatórios.
  - Genético: também treinado em mundos aleatórios, evolui uma rede reativa.

Gera as curvas de aprendizagem e guarda os modelos em resultados/.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ambientes.foraging import AmbienteForaging
from agentes.qlearning import AgenteQLearning
from aprendizagem.discretizadores import foraging_discretizador
from aprendizagem import qlearning_treino, algoritmo_genetico
from configs import foraging as cfg
from experiencias.comum import caminho, media_movel

EPISODIOS_QL = 5000


def treina_qlearning():
    print("\n##### Q-LEARNING — FORAGING #####")
    # treina num mundo aleatório (estado reativo generaliza)
    amb = AmbienteForaging(lado=12, n_recursos=8, prob_obstaculo=0.08,
                           max_passos=250, aleatorio=True)
    agente = AgenteQLearning("qlf", foraging_discretizador(),
                             taxa=0.2, desconto=0.95, epsilon=1.0)

    hist = qlearning_treino.treina(amb, agente, n_episodios=EPISODIOS_QL,
                                   epsilon_min=0.05, decaimento=0.9990)

    agente.guarda(caminho("ql_foraging.pkl"))
    # guardamos a mesma tabela para as duas dificuldades (política é geral)
    agente.guarda(caminho("ql_foraging_dif0.pkl"))
    agente.guarda(caminho("ql_foraging_dif1.pkl"))
    with open(caminho("hist_ql_foraging.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f)

    pontos = [h["score"] for h in hist]
    suav = media_movel(pontos, 100)
    plt.figure(figsize=(9, 5))
    plt.plot(range(len(suav)), suav, color="tab:blue")
    plt.xlabel("Episódio")
    plt.ylabel("Pontos depositados (média móvel)")
    plt.title("Curva de aprendizagem — Q-learning (Foraging)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho("curva_ql_foraging.png"), dpi=130)
    plt.close()


def treina_genetico():
    print("\n##### GENÉTICO — FORAGING #####")

    def fabrica():
        return AmbienteForaging(lado=12, n_recursos=8, prob_obstaculo=0.08,
                                max_passos=250, aleatorio=True)

    hist, melhor, fit = algoritmo_genetico.treina(
        fabrica, n_entradas=7, n_escondidas=16, pop_size=60, geracoes=150,
        episodios=1, taxa_mut=0.12, sigma=0.3, elite=3, alpha=0.5, seed=20)

    np.save(caminho("ga_foraging.npy"), melhor)
    with open(caminho("hist_ga_foraging.json"), "w", encoding="utf-8") as f:
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
    fig.suptitle("Curva de aprendizagem — Genético (Foraging)")
    fig.tight_layout()
    fig.savefig(caminho("curva_ga_foraging.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    treina_qlearning()
    treina_genetico()
    print("\nConcluído. Modelos e curvas em resultados/.")
