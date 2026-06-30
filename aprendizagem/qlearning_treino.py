"""
Ciclo de treino do Q-learning.

Corre vários episódios em modo de aprendizagem, com a probabilidade de
exploração (epsilon) a decrescer ao longo do tempo. Em cada episódio o agente
aprende online (atualização de Bellman dentro do simulador) e registamos a
recompensa e o nº de passos para depois desenhar a curva de aprendizagem.
"""

import numpy as np

from core.simulador import Simulador
from core.metricas import recompensa_total


def treina(ambiente, agente, n_episodios=4000, epsilon_inicial=1.0,
           epsilon_min=0.05, decaimento=0.999, verboso=True):
    sim = Simulador(ambiente)
    agente.epsilon = epsilon_inicial
    historico = []

    for ep in range(1, n_episodios + 1):
        resultado = sim.corre_episodio(agente, treino=True)

        historico.append({
            "episodio": ep,
            "recompensa": recompensa_total(resultado),
            "passos": resultado["passos"],
            "sucesso": bool(resultado["sucesso"]),
            "score": resultado["score"],
        })

        # decaimento do epsilon
        agente.epsilon = max(epsilon_min, agente.epsilon * decaimento)

        if verboso and ep % max(1, n_episodios // 10) == 0:
            ultimos = historico[-200:]
            taxa = 100.0 * np.mean([h["sucesso"] for h in ultimos])
            rec = np.mean([h["recompensa"] for h in ultimos])
            print(f"  ep {ep:5d}/{n_episodios} | epsilon {agente.epsilon:.3f} "
                  f"| sucesso(200) {taxa:5.1f}% | recompensa(200) {rec:6.2f}")

    return historico
