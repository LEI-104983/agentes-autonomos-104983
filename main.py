"""
Programa principal: menu simples para visualizar os ambientes e os agentes.

Permite escolher o problema (Labirinto ou Foraging), o tipo de agente (Fixo,
Q-learning ou Genético) e o nível de dificuldade, e ver a simulação numa janela
2D. Os agentes que aprendem precisam de ter sido treinados antes (ver pasta
experiencias/); se o modelo não existir, é usado o agente fixo.
"""

import os

from core.simulador import Simulador
from ambientes.labirinto import AmbienteLabirinto
from ambientes.foraging import AmbienteForaging
from agentes.fixo_labirinto import AgenteFixoLabirinto
from agentes.fixo_foraging import AgenteFixoForaging
from agentes.qlearning import AgenteQLearning
from agentes.genetico import AgenteGenetico
from aprendizagem.discretizadores import maze_discretizador, foraging_discretizador
from configs import labirinto as cfg_lab
from configs import foraging as cfg_for

PASTA = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(PASTA, "resultados")


# ----------------------------------------------------------------------
def cria_labirinto(dif):
    p = cfg_lab.NIVEIS[dif]
    return AmbienteLabirinto(lado=p["lado"], seed=p["seed"], max_passos=p["max_passos"])


def cria_foraging(dif):
    p = cfg_for.NIVEIS[dif]
    return AmbienteForaging(lado=p["lado"], seed=p["seed"], n_recursos=p["n_recursos"],
                            prob_obstaculo=p["prob_obstaculo"], max_passos=p["max_passos"])


def agente_labirinto(tipo, dif):
    if tipo == "ql":
        cam = os.path.join(RES, f"ql_labirinto_dif{dif}.pkl")
        if os.path.exists(cam):
            ag = AgenteQLearning("ql", maze_discretizador(cfg_lab.NIVEIS[dif]["lado"]), epsilon=0.0)
            ag.carrega_tabela(cam)
            return ag
        print("  (modelo Q-learning não encontrado, a usar agente fixo)")
    elif tipo == "ga":
        cam = os.path.join(RES, "ga_labirinto.npy")
        if os.path.exists(cam):
            return AgenteGenetico.carrega(cam, n_entradas=8)
        print("  (modelo genético não encontrado, a usar agente fixo)")
    return AgenteFixoLabirinto("fixo")


def agente_foraging(tipo, dif):
    if tipo == "ql":
        cam = os.path.join(RES, f"ql_foraging_dif{dif}.pkl")
        if os.path.exists(cam):
            ag = AgenteQLearning("ql", foraging_discretizador(), epsilon=0.0)
            ag.carrega_tabela(cam)
            return ag
        print("  (modelo Q-learning não encontrado, a usar agente fixo)")
    elif tipo == "ga":
        cam = os.path.join(RES, "ga_foraging.npy")
        if os.path.exists(cam):
            return AgenteGenetico.carrega(cam, n_entradas=7)
        print("  (modelo genético não encontrado, a usar agente fixo)")
    return AgenteFixoForaging("fixo")


# ----------------------------------------------------------------------
def visualiza(ambiente, agente):
    from visualizacao.render import Render
    render = Render()
    sim = Simulador(ambiente)
    res = sim.corre_episodio(agente, render=render)
    # desenha o estado final mais um instante
    render.desenha(ambiente, titulo=f"fim — passos={res['passos']} sucesso={res['sucesso']}")
    input("\n  (Enter para fechar a janela)")
    render.fecha()
    print(f"  Resultado: passos={res['passos']} | sucesso={res['sucesso']} | score={res['score']}")


def escolhe(msg, opcoes):
    print(msg)
    for chave, txt in opcoes.items():
        print(f"   {chave}. {txt}")
    return input("  > ").strip()


def menu():
    while True:
        op = escolhe("\n===== Simulador SMA (nº 104983) — Foraging & Labirinto =====", {
            "1": "Visualizar Labirinto",
            "2": "Visualizar Foraging",
            "0": "Sair",
        })

        if op == "0":
            print("A terminar.")
            break

        if op in ("1", "2"):
            tipo = escolhe("  Tipo de agente:", {
                "1": "Fixo (referência)", "2": "Q-learning (treinado)", "3": "Genético (treinado)"})
            tipo = {"1": "fixo", "2": "ql", "3": "ga"}.get(tipo, "fixo")

            if op == "1":
                dif = escolhe("  Dificuldade:", {"0": "0 (11x11)", "1": "1 (19x19)", "2": "2 (29x29)"})
                dif = int(dif) if dif in ("0", "1", "2") else 0
                visualiza(cria_labirinto(dif), agente_labirinto(tipo, dif))
            else:
                dif = escolhe("  Dificuldade:", {"0": "0 (12x12)", "1": "1 (16x16)"})
                dif = int(dif) if dif in ("0", "1") else 0
                visualiza(cria_foraging(dif), agente_foraging(tipo, dif))
        else:
            print("  Opção inválida.")


if __name__ == "__main__":
    menu()
