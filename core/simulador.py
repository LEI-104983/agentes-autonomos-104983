"""
Motor de simulação (Simulador).

Coordena o ciclo de tempo: a cada passo pede a observação ao ambiente,
pergunta a ação ao agente, executa-a, calcula a recompensa e avisa o agente.
Suporta os dois modos pedidos no enunciado:

  - Modo de Aprendizagem: a política do agente é alterada ao longo dos
    episódios (o treino propriamente dito é feito pelos módulos em
    aprendizagem/, que reutilizam o método corre_episodio deste simulador).
  - Modo de Teste: a política é fixa e apenas registamos as métricas.

Implementa o interface pedido: cria(), listaAgentes(), executa().
"""

import json
from core.metricas import resume


class Simulador:

    def __init__(self, ambiente, agentes=None, max_passos=None):
        self.ambiente = ambiente
        self.agentes = agentes if agentes is not None else []
        self.max_passos = max_passos or ambiente.max_passos

    # ------------------------------------------------------------------
    @classmethod
    def cria(cls, ficheiro_parametros):
        """Cria um simulador a partir de um ficheiro JSON com a descrição do
        ambiente e dos agentes. Usado sobretudo para guardar/repor cenários;
        as experiências montam o simulador diretamente em Python."""
        with open(ficheiro_parametros, "r", encoding="utf-8") as f:
            dados = json.load(f)

        # importação tardia para evitar dependências circulares
        from ambientes.labirinto import AmbienteLabirinto
        from ambientes.foraging import AmbienteForaging

        nome = dados.get("ambiente")
        if nome == "labirinto":
            amb = AmbienteLabirinto(**dados.get("parametros", {}))
        elif nome == "foraging":
            amb = AmbienteForaging(**dados.get("parametros", {}))
        else:
            raise ValueError(f"Ambiente desconhecido: {nome}")

        return cls(amb, agentes=[], max_passos=dados.get("max_passos"))

    # ------------------------------------------------------------------
    def adiciona_agente(self, agente):
        self.agentes.append(agente)

    def listaAgentes(self):
        return list(self.agentes)

    # ------------------------------------------------------------------
    def corre_episodio(self, agente, render=None, treino=False):
        """Corre um episódio completo com um agente. Devolve um dicionário com
        os dados do episódio (usado tanto para teste como para aprendizagem).

        Se treino=True, chamamos agente.aprende(...) quando esse método existe
        (é o caso do Q-learning, que aprende online)."""
        pos_inicial = self.ambiente.reset()
        self.ambiente.regista_agente(agente, pos_inicial)
        agente.reset()
        agente.posicao = pos_inicial

        obs = self.ambiente.observacaoPara(agente)
        agente.observacao(obs)

        recompensas = []
        trajetoria = [pos_inicial]
        terminou = False
        passos = 0

        while not terminou and passos < self.max_passos:
            if render is not None:
                render.desenha(self.ambiente, titulo=f"passo {passos}")

            accao = agente.age()
            pos_antiga, pos_nova, info = self.ambiente.agir(accao, agente)
            recompensa, fim = self.ambiente.compute_reward(agente, pos_antiga, pos_nova, info)

            nova_obs = self.ambiente.observacaoPara(agente)

            # aprendizagem online (Q-learning): passamos a transição completa
            if treino and hasattr(agente, "aprende"):
                agente.aprende(obs, accao, recompensa, nova_obs, fim)

            agente.avaliacaoEstadoAtual(recompensa)
            agente.observacao(nova_obs)
            obs = nova_obs

            recompensas.append(recompensa)
            trajetoria.append(pos_nova)

            self.ambiente.atualizacao()
            passos += 1
            terminou = fim or self.ambiente.terminou()

        sucesso = getattr(self.ambiente, "sucesso", lambda: False)()
        score = getattr(self.ambiente, "valor_objetivo", lambda: float(sucesso))()

        return {
            "passos": passos,
            "recompensas": recompensas,
            "trajetoria": trajetoria,
            "sucesso": sucesso,
            "score": score,
        }

    # ------------------------------------------------------------------
    def executa(self, n_episodios=1, agente=None, render=None, gama=0.95, verboso=False):
        """Modo de Teste: corre vários episódios com política fixa e devolve as
        métricas agregadas."""
        if agente is None:
            if not self.agentes:
                raise ValueError("Não há agentes registados no simulador.")
            agente = self.agentes[0]

        episodios = []
        for i in range(n_episodios):
            ep = self.corre_episodio(agente, render=render, treino=False)
            episodios.append(ep)
            if verboso:
                print(f"  episódio {i + 1}/{n_episodios}: "
                      f"passos={ep['passos']} sucesso={ep['sucesso']}")

        return resume(episodios, gama=gama), episodios
