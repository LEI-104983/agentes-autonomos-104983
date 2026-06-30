"""
Parâmetros do ambiente Labirinto.

Definimos três níveis de dificuldade, com labirintos progressivamente maiores.
As dimensões são ímpares por causa do algoritmo de geração (paredes entre
células). As seeds garantem que os mapas são reprodutíveis.
"""

# dificuldade -> (lado do labirinto, seed do mapa, máximo de passos)
NIVEIS = {
    0: {"lado": 11, "seed": 7,  "max_passos": 120},
    1: {"lado": 19, "seed": 21, "max_passos": 350},
    2: {"lado": 29, "seed": 33, "max_passos": 800},
}

# recompensas (reward shaping)
REC_PASSO = -0.05        # custo por passo, para incentivar caminhos curtos
REC_COLISAO = -0.5       # penalização por bater numa parede
REC_APROXIMA = 0.1       # bónus por reduzir a distância ao objetivo
REC_AFASTA = -0.15       # penalização por se afastar
REC_OBJETIVO = 10.0      # recompensa por chegar à saída
BONUS_TEMPO = 5.0        # bónus extra que decresce com o nº de passos usados
