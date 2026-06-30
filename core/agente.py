"""
Classe abstrata Agente.

Define a interface comum a todos os agentes do simulador, seguindo os métodos
pedidos no enunciado (cria, observacao, age, avaliacaoEstadoAtual, instala,
comunica). As subclasses concretas (fixo, q-learning, genetico) implementam o
método age(), que é onde está a "inteligência" de cada estratégia.
"""

from abc import ABC, abstractmethod
import json


class Agente(ABC):
    # ações possíveis: 0=cima, 1=direita, 2=baixo, 3=esquerda
    N_ACOES = 4

    def __init__(self, id_agente, alcance_sensor=1):
        self.id = id_agente
        self.alcance_sensor = alcance_sensor

        # estado interno usado durante um episódio
        self.posicao = None
        self.ultima_obs = None
        self.recompensas = []          # recompensas recebidas no episódio atual
        self.caixa_mensagens = []      # mensagens recebidas de outros agentes

    # ------------------------------------------------------------------
    # Construção a partir de um ficheiro de parâmetros (JSON)
    # ------------------------------------------------------------------
    @classmethod
    def cria(cls, ficheiro_parametros):
        with open(ficheiro_parametros, "r", encoding="utf-8") as f:
            dados = json.load(f)
        return cls(
            id_agente=dados.get("id", "agente"),
            alcance_sensor=dados.get("alcance_sensor", 1),
        )

    # ------------------------------------------------------------------
    # Perceção / atuação
    # ------------------------------------------------------------------
    def observacao(self, obs):
        """Recebe a observação que o ambiente preparou para este agente."""
        self.ultima_obs = obs

    @abstractmethod
    def age(self):
        """Escolhe e devolve o índice da próxima ação (0..3)."""
        raise NotImplementedError

    def avaliacaoEstadoAtual(self, recompensa):
        """Recebe a recompensa do último passo. As subclasses que aprendem
        podem estender este método para atualizar a sua política."""
        self.recompensas.append(recompensa)

    # ------------------------------------------------------------------
    # Métodos de interface (incluídos para o interface ficar completo)
    # ------------------------------------------------------------------
    def instala(self, sensor):
        """Permite trocar/ajustar o alcance do sensor do agente."""
        self.alcance_sensor = sensor

    def comunica(self, mensagem, de_agente):
        """Recebe uma mensagem de outro agente. Não usamos comunicação
        explícita nos cenários atuais, mas o método existe para suportar
        extensões cooperativas futuras."""
        self.caixa_mensagens.append((de_agente, mensagem))

    # ------------------------------------------------------------------
    def reset(self):
        """Limpa o estado interno entre episódios."""
        self.recompensas = []
        self.caixa_mensagens = []
        self.ultima_obs = None
