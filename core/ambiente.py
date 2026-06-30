"""
Classe abstrata Ambiente.

Representa o mundo de simulação (uma grelha 2D) e implementa a interface base
pedida no enunciado: observacaoPara, atualizacao e agir. A lógica que muda de
problema para problema (como é gerado o mapa, como se calcula a recompensa e
que observação é dada ao agente) fica nos métodos abstratos, implementados
pelas subclasses AmbienteLabirinto e AmbienteForaging.

Convenção de coordenadas: as posições são (linha, coluna) e o mapa é um array
NumPy mapa[linha, coluna].
"""

from abc import ABC, abstractmethod
import numpy as np

# tipos de célula
VAZIO = 0
PAREDE = 1
OBJETIVO = 2     # saída do labirinto
NINHO = 3        # ponto de entrega no foraging
RECURSO = 4      # recurso a recolher no foraging

# 0=cima, 1=direita, 2=baixo, 3=esquerda
ACAO_DELTA = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}


class Ambiente(ABC):

    def __init__(self, n_linhas, n_colunas, max_passos=200):
        self.n_linhas = n_linhas
        self.n_colunas = n_colunas
        self.max_passos = max_passos

        self.mapa = None
        self.passos = 0
        self.agentes = []
        self.posicoes = {}      # id_agente -> (linha, coluna)

    # ------------------------------------------------------------------
    # Métodos que cada problema tem de definir
    # ------------------------------------------------------------------
    @abstractmethod
    def gera_mapa(self):
        """Constrói self.mapa e devolve a posição inicial dos agentes."""
        raise NotImplementedError

    @abstractmethod
    def observacaoPara(self, agente):
        """Devolve a observação (vetor NumPy) para o agente dado."""
        raise NotImplementedError

    @abstractmethod
    def compute_reward(self, agente, pos_antiga, pos_nova, info):
        """Devolve (recompensa, terminou) para a transição efetuada."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Registo de agentes
    # ------------------------------------------------------------------
    def regista_agente(self, agente, pos_inicial):
        if agente not in self.agentes:
            self.agentes.append(agente)
        self.posicoes[agente.id] = pos_inicial
        agente.posicao = pos_inicial

    def posicao_de(self, agente):
        return self.posicoes.get(agente.id)

    # ------------------------------------------------------------------
    # Ajudas de grelha
    # ------------------------------------------------------------------
    def dentro(self, linha, coluna):
        return 0 <= linha < self.n_linhas and 0 <= coluna < self.n_colunas

    def e_livre(self, linha, coluna):
        """Verdadeiro se a célula existe e não é parede."""
        return self.dentro(linha, coluna) and self.mapa[linha, coluna] != PAREDE

    # ------------------------------------------------------------------
    # Movimento (comum a todos os ambientes)
    # ------------------------------------------------------------------
    def agir(self, accao, agente):
        """Tenta mover o agente segundo a ação. Devolve (pos_antiga, pos_nova,
        info). Quem calcula a recompensa é compute_reward, chamado pelo
        simulador a seguir."""
        pos_antiga = self.posicoes[agente.id]
        l, c = pos_antiga
        dl, dc = ACAO_DELTA[accao]
        nl, nc = l + dl, c + dc

        info = {"colisao": False}
        pos_nova = pos_antiga

        if not self.e_livre(nl, nc):
            info["colisao"] = True          # bateu numa parede ou na borda
        else:
            pos_nova = (nl, nc)
            self.posicoes[agente.id] = pos_nova
            agente.posicao = pos_nova

        return pos_antiga, pos_nova, info

    # ------------------------------------------------------------------
    # Ciclo de tempo
    # ------------------------------------------------------------------
    def atualizacao(self):
        """Avança o relógio do ambiente um passo. Subclasses com dinâmica
        própria (ex.: recursos que reaparecem) podem estender este método."""
        self.passos += 1

    @abstractmethod
    def terminou(self):
        """Indica se o episódio terminou."""
        raise NotImplementedError

    def reset(self):
        """Repõe o ambiente para o início de um episódio."""
        self.passos = 0
        self.posicoes = {}
        pos_inicial = self.gera_mapa()
        return pos_inicial
