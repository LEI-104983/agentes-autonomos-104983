"""
Ambiente de Recoleção (Foraging).

Uma grelha 2D com obstáculos, um ninho (ponto de entrega) e vários recursos
espalhados, cada um com um valor. O agente move-se, apanha um recurso de cada
vez e tem de o levar ao ninho para o valor contar. O objetivo do sistema é
maximizar o total de pontos depositados dentro do limite de tempo.

Perceção do agente: conteúdo das 4 células vizinhas, se está a transportar um
recurso, e a direção para o "alvo atual" (o recurso mais próximo quando tem as
mãos livres, ou o ninho quando está carregado).
"""

import numpy as np

from core.ambiente import Ambiente, PAREDE, VAZIO, NINHO, RECURSO, ACAO_DELTA
from configs import foraging as cfg


class AmbienteForaging(Ambiente):

    def __init__(self, lado=12, seed=5, n_recursos=8, prob_obstaculo=0.08,
                 max_passos=200, aleatorio=False):
        super().__init__(lado, lado, max_passos)
        self.lado = lado
        self.seed = seed
        self.n_recursos = n_recursos
        self.prob_obstaculo = prob_obstaculo
        self.aleatorio = aleatorio

        self.ninho = (lado // 2, lado // 2)
        self.inicio = self.ninho

        # estado do episódio
        self.valores = {}              # posição -> valor do recurso
        self.transporta = 0            # valor do recurso transportado (0 = nada)
        self.pontos_depositados = 0

        if not aleatorio:
            self._base_mapa, self._base_valores = self._constroi(np.random.default_rng(seed))
        else:
            self._base_mapa = None

    # ------------------------------------------------------------------
    def _constroi(self, rng):
        mapa = np.zeros((self.lado, self.lado), dtype=int)

        # bordas são parede
        mapa[0, :] = PAREDE
        mapa[-1, :] = PAREDE
        mapa[:, 0] = PAREDE
        mapa[:, -1] = PAREDE

        # obstáculos interiores aleatórios
        for l in range(1, self.lado - 1):
            for c in range(1, self.lado - 1):
                if rng.random() < self.prob_obstaculo:
                    mapa[l, c] = PAREDE

        # garantir que o ninho está livre
        mapa[self.ninho] = NINHO

        # colocar recursos em células livres
        valores = {}
        livres = [(l, c) for l in range(1, self.lado - 1) for c in range(1, self.lado - 1)
                  if mapa[l, c] == VAZIO and (l, c) != self.ninho]
        rng.shuffle(livres)
        for pos in livres[:self.n_recursos]:
            valor = cfg.VALORES_RECURSO[rng.integers(len(cfg.VALORES_RECURSO))]
            mapa[pos] = RECURSO
            valores[pos] = valor

        return mapa, valores

    # ------------------------------------------------------------------
    def gera_mapa(self):
        if self.aleatorio:
            rng = np.random.default_rng(np.random.randint(0, 1_000_000))
            self.mapa, self.valores = self._constroi(rng)
        else:
            self.mapa = self._base_mapa.copy()
            self.valores = dict(self._base_valores)

        self.transporta = 0
        self.pontos_depositados = 0
        return self.inicio

    # ------------------------------------------------------------------
    def _alvo_atual(self):
        """Ninho se estiver carregado; senão o recurso mais próximo."""
        if self.transporta > 0 or not self.valores:
            return self.ninho
        return min(self.valores.keys(), key=lambda p: self._dist(p, self._pos_agente))

    @property
    def _pos_agente(self):
        # só há um agente nestes cenários
        return next(iter(self.posicoes.values())) if self.posicoes else self.inicio

    @staticmethod
    def _dist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _codigo_celula(self, l, c):
        if not self.dentro(l, c) or self.mapa[l, c] == PAREDE:
            return -1.0
        if self.mapa[l, c] == RECURSO:
            return 0.5
        if self.mapa[l, c] == NINHO:
            return 1.0
        return 0.0

    def observacaoPara(self, agente):
        l, c = self.posicoes[agente.id]
        obs = np.zeros(7, dtype=np.float32)

        for i, accao in enumerate(range(4)):
            dl, dc = ACAO_DELTA[accao]
            obs[i] = self._codigo_celula(l + dl, c + dc)

        obs[4] = 1.0 if self.transporta > 0 else 0.0

        alvo = self._alvo_atual()
        obs[5] = (alvo[0] - l) / (self.lado - 1)
        obs[6] = (alvo[1] - c) / (self.lado - 1)
        return obs

    # ------------------------------------------------------------------
    def compute_reward(self, agente, pos_antiga, pos_nova, info):
        if info["colisao"]:
            return cfg.REC_COLISAO + cfg.REC_PASSO, False

        recompensa = cfg.REC_PASSO

        # reward shaping em relação ao alvo (calculado a partir da posição antiga)
        alvo = self.ninho if self.transporta > 0 else (
            min(self.valores.keys(), key=lambda p: self._dist(p, pos_antiga))
            if self.valores else self.ninho)
        if self._dist(pos_nova, alvo) < self._dist(pos_antiga, alvo):
            recompensa += cfg.REC_APROXIMA
        else:
            recompensa += cfg.REC_AFASTA

        # apanhar um recurso
        if self.transporta == 0 and pos_nova in self.valores:
            self.transporta = self.valores.pop(pos_nova)
            self.mapa[pos_nova] = VAZIO
            recompensa += cfg.REC_RECOLHE

        # depositar no ninho
        elif self.transporta > 0 and pos_nova == self.ninho:
            recompensa += self.transporta * cfg.REC_DEPOSITA
            self.pontos_depositados += self.transporta
            self.transporta = 0

        done = (not self.valores) and self.transporta == 0
        return recompensa, done

    # ------------------------------------------------------------------
    def terminou(self):
        if self.passos >= self.max_passos:
            return True
        return (not self.valores) and self.transporta == 0

    def sucesso(self):
        """Considera-se sucesso ter limpo o mapa (todos os recursos entregues)."""
        return (not self.valores) and self.transporta == 0

    def valor_objetivo(self):
        return float(self.pontos_depositados)
