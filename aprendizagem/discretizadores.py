"""
Discretizadores: transformam a observação contínua de cada ambiente num estado
discreto (tuplo) para o Q-learning tabular.

Cada ambiente tem necessidades diferentes:
  - No labirinto recuperamos a posição exata do agente a partir da observação
    normalizada. Assim o Q-learning "memoriza" o caminho do labirinto em que é
    treinado (muito eficiente nesse mapa, mas não generaliza para outros).
  - No foraging usamos um estado reativo: paredes à volta, se transporta um
    recurso e a direção (em sinal) para o alvo atual.
"""

import numpy as np


def maze_discretizador(lado):
    def discretiza(obs):
        l = int(round(obs[4] * (lado - 1)))
        c = int(round(obs[5] * (lado - 1)))
        return (l, c)
    return discretiza


def foraging_discretizador():
    def sinal(v):
        if v > 0.05:
            return 1
        if v < -0.05:
            return -1
        return 0

    def discretiza(obs):
        paredes = tuple(1 if obs[i] == -1.0 else 0 for i in range(4))
        carrega = int(obs[4])
        return paredes + (carrega, sinal(obs[5]), sinal(obs[6]))
    return discretiza
