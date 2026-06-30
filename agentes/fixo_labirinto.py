"""
Agente de política fixa para o labirinto: seguidor de parede (regra da mão
direita). Mantém uma "orientação" interna e, em cada passo, tenta virar à
direita; se houver parede segue em frente, depois à esquerda e, em último caso,
volta para trás. Num labirinto perfeito esta regra garante que se chega sempre
à saída (embora possa percorrer muitos corredores).

Não aprende: serve de referência para comparar com os agentes que aprendem.
"""

import numpy as np

from core.agente import Agente

# 0=cima, 1=direita, 2=baixo, 3=esquerda
DIREITA = 1
ESQUERDA = -1


class AgenteFixoLabirinto(Agente):

    def __init__(self, id_agente="fixo", alcance_sensor=1):
        super().__init__(id_agente, alcance_sensor)
        self.orientacao = 2          # começa virado para baixo (rumo à saída)

    def reset(self):
        super().reset()
        self.orientacao = 2

    def age(self):
        if self.ultima_obs is None:
            return self.orientacao

        livre = self.ultima_obs[:4]   # 1 se a célula nessa direção está livre

        # ordem da regra da mão direita: direita, frente, esquerda, trás
        tentativas = [
            (self.orientacao + DIREITA) % 4,
            self.orientacao,
            (self.orientacao + ESQUERDA) % 4,
            (self.orientacao + 2) % 4,
        ]
        for direcao in tentativas:
            if livre[direcao] == 1.0:
                self.orientacao = direcao
                return direcao

        # encurralado (não devia acontecer num labirinto perfeito)
        return self.orientacao
