"""
Visualização 2D dos ambientes com matplotlib.

Desenha a grelha com cores por tipo de célula (vazio, parede, objetivo, ninho,
recurso) e marca a posição dos agentes. Serve tanto para acompanhar uma
simulação em tempo real (janela animada) como para gravar imagens para o
relatório.
"""

import matplotlib
import numpy as np

from core.ambiente import VAZIO, PAREDE, OBJETIVO, NINHO, RECURSO

# cores por valor de célula (índices 0..4)
CORES = ["#ffffff", "#2b2b2b", "#3cb371", "#4169e1", "#ffb300"]
LEGENDA = {
    VAZIO: "vazio", PAREDE: "parede", OBJETIVO: "objetivo",
    NINHO: "ninho", RECURSO: "recurso",
}


def _desenha_no_eixo(ax, ambiente, titulo=""):
    from matplotlib.colors import ListedColormap

    ax.clear()
    cmap = ListedColormap(CORES)
    ax.imshow(ambiente.mapa, cmap=cmap, vmin=0, vmax=4)

    # agentes (vermelho)
    for pos in ambiente.posicoes.values():
        ax.scatter(pos[1], pos[0], c="red", s=120, marker="o",
                   edgecolors="black", zorder=3)

    ax.set_title(titulo)
    ax.set_xticks([])
    ax.set_yticks([])


class Render:
    """Janela animada para acompanhar uma simulação."""

    def __init__(self, intervalo=0.06):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.intervalo = intervalo
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(5, 5))

    def desenha(self, ambiente, titulo=""):
        _desenha_no_eixo(self.ax, ambiente, titulo)
        self.fig.canvas.draw_idle()
        self.plt.pause(self.intervalo)

    def fecha(self):
        self.plt.ioff()
        self.plt.close(self.fig)


def grava_imagem(ambiente, ficheiro, titulo=""):
    """Grava uma imagem do estado atual do ambiente (usa backend Agg)."""
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    _desenha_no_eixo(ax, ambiente, titulo)
    fig.tight_layout()
    fig.savefig(ficheiro, dpi=130)
    plt.close(fig)
