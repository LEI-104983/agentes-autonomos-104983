"""
Parâmetros do ambiente de Recoleção (Foraging).

O mundo é uma grelha com obstáculos, um ninho (ponto de entrega) e vários
recursos com valores diferentes. O agente recolhe um recurso de cada vez e
tem de o levar até ao ninho para o ponto contar.
"""

# dificuldade -> configuração do mundo
NIVEIS = {
    0: {"lado": 12, "seed": 5,  "n_recursos": 8,  "prob_obstaculo": 0.08, "max_passos": 200},
    1: {"lado": 16, "seed": 14, "n_recursos": 12, "prob_obstaculo": 0.12, "max_passos": 300},
}

# valores possíveis dos recursos (escolhidos aleatoriamente para cada recurso)
VALORES_RECURSO = [1, 1, 1, 2, 3]

# recompensas
REC_PASSO = -0.02        # pequeno custo por passo
REC_COLISAO = -0.2       # bater numa parede
REC_RECOLHE = 1.0        # apanhar um recurso
REC_DEPOSITA = 2.0       # multiplicador aplicado ao valor do recurso depositado
REC_APROXIMA = 0.05      # aproximar-se do alvo atual (recurso ou ninho)
REC_AFASTA = -0.05       # afastar-se do alvo atual
