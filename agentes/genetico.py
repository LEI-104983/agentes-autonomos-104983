"""
Agente genético.

A sua política é uma rede neuronal feedforward (ver aprendizagem/rede.py) cujos
pesos foram evoluídos por um algoritmo genético. Em execução o agente não
aprende: limita-se a aplicar o genoma que recebeu. O treino é feito à parte,
em aprendizagem/algoritmo_genetico.py.
"""

import numpy as np

from core.agente import Agente
from aprendizagem.rede import RedePolitica


class AgenteGenetico(Agente):

    def __init__(self, id_agente="genetico", n_entradas=8, n_escondidas=16,
                 genoma=None, alcance_sensor=1):
        super().__init__(id_agente, alcance_sensor)
        self.rede = RedePolitica(n_entradas, n_escondidas, Agente.N_ACOES)
        if genoma is not None:
            self.rede.carrega_vetor(np.asarray(genoma, dtype=np.float64))

    def define_genoma(self, genoma):
        self.rede.carrega_vetor(np.asarray(genoma, dtype=np.float64))

    def age(self):
        if self.ultima_obs is None:
            return np.random.randint(Agente.N_ACOES)
        return self.rede.melhor_acao(self.ultima_obs)

    # ------------------------------------------------------------------
    @classmethod
    def carrega(cls, ficheiro_npy, n_entradas, n_escondidas=16, id_agente="genetico"):
        genoma = np.load(ficheiro_npy)
        return cls(id_agente, n_entradas, n_escondidas, genoma=genoma)
