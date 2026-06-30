"""
Agente de política fixa para o foraging: heurística gulosa.

Move-se sempre na direção do alvo atual (o recurso mais próximo quando tem as
mãos livres, ou o ninho quando está carregado). Essa direção é dada pela
observação. Se a célula preferida estiver bloqueada por uma parede, tenta a
direção alternativa e, em último caso, escolhe uma vizinha livre ao acaso para
se desencravar.

É uma política simples e não ótima (pode ficar presa atrás de obstáculos), o
que a torna uma boa referência de comparação.
"""

import numpy as np

from core.agente import Agente


class AgenteFixoForaging(Agente):

    def __init__(self, id_agente="fixo", alcance_sensor=1):
        super().__init__(id_agente, alcance_sensor)

    def reset(self):
        super().reset()

    def age(self):
        obs = self.ultima_obs
        if obs is None:
            return np.random.randint(Agente.N_ACOES)

        paredes_livres = obs[:4]          # 1 livre? (na verdade código da célula)
        livre = [obs[i] != -1.0 for i in range(4)]
        dir_l, dir_c = obs[5], obs[6]     # direção (com sinal) para o alvo

        # direções desejadas, da mais útil para a menos útil
        desejadas = []
        vertical = 2 if dir_l > 0 else 0          # baixo / cima
        horizontal = 1 if dir_c > 0 else 3        # direita / esquerda

        if abs(dir_l) >= abs(dir_c):
            desejadas = [vertical, horizontal]
        else:
            desejadas = [horizontal, vertical]

        for direcao in desejadas:
            if abs(dir_l if direcao in (0, 2) else dir_c) > 0 and livre[direcao]:
                return direcao

        # nenhuma direção útil disponível -> escolher uma vizinha livre ao acaso
        opcoes = [d for d in range(4) if livre[d]]
        if opcoes:
            return int(np.random.choice(opcoes))
        return np.random.randint(Agente.N_ACOES)
