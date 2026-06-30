"""
Agente Q-learning (tabular).

Mantém uma tabela Q indexada por estados discretos. Como as observações dos
ambientes são contínuas, recebe um "discretizador" — uma função que transforma
o vetor de observação num estado discreto (um tuplo). A tabela é guardada num
dicionário, por isso só ocupamos memória com os estados realmente visitados.

Suporta os dois modos: em aprendizagem usa ε-greedy e atualiza a tabela; em
teste usa ε=0 (sempre a melhor ação conhecida).
"""

import numpy as np

from core.agente import Agente


class AgenteQLearning(Agente):

    def __init__(self, id_agente, discretizador, taxa=0.2, desconto=0.95,
                 epsilon=1.0, alcance_sensor=1):
        super().__init__(id_agente, alcance_sensor)
        self.discretizador = discretizador
        self.taxa = taxa
        self.desconto = desconto
        self.epsilon = epsilon
        self.Q = {}                 # estado (tuplo) -> np.array com 4 valores

    # ------------------------------------------------------------------
    def _valores(self, estado):
        if estado not in self.Q:
            self.Q[estado] = np.zeros(Agente.N_ACOES)
        return self.Q[estado]

    def age(self):
        if self.ultima_obs is None:
            return np.random.randint(Agente.N_ACOES)
        estado = self.discretizador(self.ultima_obs)
        if np.random.rand() < self.epsilon:
            return np.random.randint(Agente.N_ACOES)
        return int(np.argmax(self._valores(estado)))

    # ------------------------------------------------------------------
    def aprende(self, obs, accao, recompensa, nova_obs, terminou):
        """Atualização de Bellman."""
        estado = self.discretizador(obs)
        novo_estado = self.discretizador(nova_obs)

        q = self._valores(estado)
        alvo = recompensa
        if not terminou:
            alvo += self.desconto * np.max(self._valores(novo_estado))
        q[accao] += self.taxa * (alvo - q[accao])

    # ------------------------------------------------------------------
    def guarda(self, ficheiro):
        import pickle
        with open(ficheiro, "wb") as f:
            pickle.dump(self.Q, f)

    def carrega_tabela(self, ficheiro):
        import pickle
        with open(ficheiro, "rb") as f:
            self.Q = pickle.load(f)
