"""
Rede neuronal feedforward simples (sem recorrência), implementada em NumPy.

É a política do agente genético: recebe o vetor de observação e devolve um
valor por ação; a ação escolhida é a do maior valor. Os pesos da rede são o
"genoma" que o algoritmo genético evolui. Optámos por uma rede feedforward (em
vez de recorrente) por ser mais simples de evoluir e suficiente para políticas
reativas.
"""

import numpy as np


class RedePolitica:

    def __init__(self, n_entradas, n_escondidas=16, n_saidas=4, rng=None):
        rng = rng or np.random.default_rng()
        self.n_entradas = n_entradas
        self.n_escondidas = n_escondidas
        self.n_saidas = n_saidas

        # pesos iniciais pequenos
        self.W1 = rng.normal(0, 0.5, size=(n_entradas, n_escondidas))
        self.b1 = np.zeros(n_escondidas)
        self.W2 = rng.normal(0, 0.5, size=(n_escondidas, n_saidas))
        self.b2 = np.zeros(n_saidas)

    # ------------------------------------------------------------------
    def avalia(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def melhor_acao(self, x):
        return int(np.argmax(self.avalia(x)))

    # ------------------------------------------------------------------
    # Conversão pesos <-> vetor (genoma)
    # ------------------------------------------------------------------
    @property
    def n_pesos(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def para_vetor(self):
        return np.concatenate([self.W1.ravel(), self.b1,
                               self.W2.ravel(), self.b2])

    def carrega_vetor(self, vetor):
        i = 0
        n = self.W1.size
        self.W1 = vetor[i:i + n].reshape(self.W1.shape); i += n
        n = self.b1.size
        self.b1 = vetor[i:i + n].copy(); i += n
        n = self.W2.size
        self.W2 = vetor[i:i + n].reshape(self.W2.shape); i += n
        n = self.b2.size
        self.b2 = vetor[i:i + n].copy(); i += n
