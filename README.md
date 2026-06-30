# Simulador de SMA — Recoleção e Labirinto

Projeto da unidade curricular de **Agentes Autónomos** (ISCTE, 2025/26).
Trabalho individual — nº **104983**.

Repositório no GitHub: https://github.com/LEI-104983/agentes-autonomos-104983

## Descrição

Simulador modular de sistemas multi-agente em Python que implementa dois dos
problemas clássicos do enunciado — **Recoleção (Foraging)** e **Labirinto** — e
compara três estratégias de agente em cada um:

- **Agente fixo** (sem aprendizagem): seguidor de parede no labirinto, heurística
  gulosa na recoleção. Serve de referência.
- **Q-learning** (aprendizagem por reforço tabular).
- **Genético**: a política é uma rede neuronal *feedforward* cujos pesos são
  evoluídos por um algoritmo genético com *novelty search*.

Há dois modos de funcionamento: **Aprendizagem** (a política é treinada e
regista-se a curva de aprendizagem) e **Teste** (política fixa, mede-se taxa de
sucesso, passos médios e recompensa). Inclui ainda um **modo de visualização 2D**
em Matplotlib.

## Estrutura

```
core/            classes abstratas Ambiente e Agente, Simulador e métricas
ambientes/       AmbienteLabirinto e AmbienteForaging
agentes/         agentes fixo, Q-learning e genético
aprendizagem/    rede neuronal, algoritmo genético, treino de Q-learning, discretizadores
visualizacao/    desenho da grelha em Matplotlib
configs/         parâmetros de cada ambiente
experiencias/    scripts de treino e de comparação (modo de teste)
resultados/      modelos treinados, curvas e gráficos de comparação
main.py          menu de visualização
```

## Instalação

```bash
pip install -r requirements.txt
```

Requer Python 3.9 ou superior (testado em 3.12).

## Como correr

**Visualizar** um agente num ambiente (janela 2D):

```bash
python main.py
```

**Treinar** os agentes (gera modelos e curvas em `resultados/`):

```bash
python -m experiencias.treinar_labirinto
python -m experiencias.treinar_foraging
```

**Comparar** os agentes em modo de teste (gera os gráficos de comparação):

```bash
python -m experiencias.comparar
```

> Nota: os modelos já treinados estão incluídos em `resultados/`, por isso é
> possível correr a visualização e a comparação sem treinar de novo.

## Interfaces (enunciado)

- `Simulador`: `cria()`, `listaAgentes()`, `executa()`
- `Ambiente`: `observacaoPara()`, `agir()`, `atualizacao()`
- `Agente`: `cria()`, `observacao()`, `age()`, `avaliacaoEstadoAtual()`,
  `instala()`, `comunica()`

A interação entre agentes e ambiente faz-se exclusivamente através destes
interfaces, o que permite, em princípio, trocar ambientes ou agentes sem alterar
o resto do sistema.
