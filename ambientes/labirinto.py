"""
Ambiente Labirinto (Maze).

Uma grelha 2D com paredes, um ponto de partida (canto superior esquerdo) e um
ponto de chegada (canto inferior direito). O labirinto é gerado por
backtracking recursivo, o que produz um labirinto "perfeito" (existe sempre um
e um só caminho entre quaisquer duas células livres).

Perceção do agente: visão local das 4 células vizinhas (livre/parede) mais a
sua posição e a direção para o objetivo (normalizadas). É uma perceção mista:
local quanto a obstáculos, global quanto à orientação.
"""

import numpy as np

from core.ambiente import Ambiente, PAREDE, VAZIO, OBJETIVO, ACAO_DELTA
from configs import labirinto as cfg


def gera_labirinto(lado, rng):
    """Backtracking recursivo. As células de passagem ficam em coordenadas
    ímpares; as paredes entre elas são abertas à medida que escavamos."""
    mapa = np.ones((lado, lado), dtype=int)  # começa tudo a parede

    def vizinhos(r, c):
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if 1 <= nr < lado - 1 and 1 <= nc < lado - 1:
                yield nr, nc, dr, dc

    inicio = (1, 1)
    mapa[inicio] = VAZIO
    visitadas = {inicio}
    pilha = [inicio]

    while pilha:
        r, c = pilha[-1]
        opcoes = [(nr, nc, dr, dc) for nr, nc, dr, dc in vizinhos(r, c)
                  if (nr, nc) not in visitadas]
        if not opcoes:
            pilha.pop()
            continue
        nr, nc, dr, dc = opcoes[rng.integers(len(opcoes))]
        mapa[r + dr // 2, c + dc // 2] = VAZIO   # abre a parede pelo meio
        mapa[nr, nc] = VAZIO
        visitadas.add((nr, nc))
        pilha.append((nr, nc))

    return mapa


class AmbienteLabirinto(Ambiente):

    def __init__(self, lado=11, seed=7, max_passos=120, aleatorio=False):
        super().__init__(lado, lado, max_passos)
        self.lado = lado
        self.seed = seed
        self.aleatorio = aleatorio   # se True, gera um mapa novo a cada reset

        self.inicio = (1, 1)
        self.objetivo = (lado - 2, lado - 2)

        # quando o mapa é fixo, geramos já aqui e reutilizamos
        if not aleatorio:
            self._mapa_base = gera_labirinto(lado, np.random.default_rng(seed))
        else:
            self._mapa_base = None

    # ------------------------------------------------------------------
    def gera_mapa(self):
        if self.aleatorio:
            seed = np.random.randint(0, 1_000_000)
            self.mapa = gera_labirinto(self.lado, np.random.default_rng(seed))
        else:
            self.mapa = self._mapa_base.copy()

        self.mapa[self.objetivo] = OBJETIVO
        return self.inicio

    # ------------------------------------------------------------------
    def _distancia(self, pos):
        return abs(pos[0] - self.objetivo[0]) + abs(pos[1] - self.objetivo[1])

    def observacaoPara(self, agente):
        l, c = self.posicoes[agente.id]
        obs = np.zeros(8, dtype=np.float32)

        # 4 vizinhos: 1 se livre, 0 se parede/borda
        for i, accao in enumerate(range(4)):
            dl, dc = ACAO_DELTA[accao]
            obs[i] = 1.0 if self.e_livre(l + dl, c + dc) else 0.0

        # posição normalizada
        obs[4] = l / (self.lado - 1)
        obs[5] = c / (self.lado - 1)
        # direção (com sinal) para o objetivo
        obs[6] = (self.objetivo[0] - l) / (self.lado - 1)
        obs[7] = (self.objetivo[1] - c) / (self.lado - 1)
        return obs

    # ------------------------------------------------------------------
    def compute_reward(self, agente, pos_antiga, pos_nova, info):
        if info["colisao"]:
            return cfg.REC_COLISAO + cfg.REC_PASSO, False

        recompensa = cfg.REC_PASSO
        if self._distancia(pos_nova) < self._distancia(pos_antiga):
            recompensa += cfg.REC_APROXIMA
        else:
            recompensa += cfg.REC_AFASTA

        if pos_nova == self.objetivo:
            bonus = cfg.BONUS_TEMPO * (1.0 - self.passos / self.max_passos)
            recompensa += cfg.REC_OBJETIVO + max(0.0, bonus)
            return recompensa, True

        return recompensa, False

    # ------------------------------------------------------------------
    def terminou(self):
        if self.passos >= self.max_passos:
            return True
        return any(pos == self.objetivo for pos in self.posicoes.values())

    def sucesso(self):
        return any(pos == self.objetivo for pos in self.posicoes.values())

    def valor_objetivo(self):
        return 1.0 if self.sucesso() else 0.0
