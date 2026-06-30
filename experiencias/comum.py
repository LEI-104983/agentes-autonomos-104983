"""Funções e caminhos partilhados pelas experiências."""

import os
import numpy as np

PASTA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PASTA, "resultados")
os.makedirs(RES, exist_ok=True)


def caminho(nome):
    return os.path.join(RES, nome)


def media_movel(valores, janela=50):
    valores = np.asarray(valores, dtype=np.float64)
    if len(valores) < janela:
        janela = max(1, len(valores) // 5 or 1)
    nucleo = np.ones(janela) / janela
    return np.convolve(valores, nucleo, mode="valid")
