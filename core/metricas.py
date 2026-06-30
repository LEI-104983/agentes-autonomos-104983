"""
Cálculo das métricas de desempenho pedidas no enunciado.

Trabalhamos sobre uma lista de "resultados de episódio", cada um um dicionário
com pelo menos as chaves:
    - "passos": nº de ações até terminar
    - "recompensas": lista das recompensas recebidas passo a passo
    - "sucesso": bool (chegou ao objetivo / depositou recursos)
"""

import numpy as np


def recompensa_total(episodio):
    return float(np.sum(episodio["recompensas"]))


def recompensa_descontada(episodio, gama=0.95):
    """Soma das recompensas descontadas: sum_t gama^t * r_t."""
    total = 0.0
    desconto = 1.0
    for r in episodio["recompensas"]:
        total += desconto * r
        desconto *= gama
    return float(total)


def resume(episodios, gama=0.95):
    """Agrega uma lista de episódios nas métricas de sistema."""
    n = len(episodios)
    if n == 0:
        return {}

    passos = [e["passos"] for e in episodios]
    sucessos = [1 if e["sucesso"] else 0 for e in episodios]
    rec_media = [recompensa_total(e) for e in episodios]
    rec_desc = [recompensa_descontada(e, gama) for e in episodios]

    # passos médios contados apenas nos episódios bem sucedidos (mais honesto
    # para comparar eficiência; se nunca tiver sucesso fica NaN)
    passos_sucesso = [p for p, s in zip(passos, sucessos) if s == 1]

    return {
        "n_episodios": n,
        "taxa_sucesso": 100.0 * np.mean(sucessos),
        "passos_medios": float(np.mean(passos)),
        "passos_medios_sucesso": float(np.mean(passos_sucesso)) if passos_sucesso else float("nan"),
        "desvio_passos": float(np.std(passos)),
        "recompensa_media": float(np.mean(rec_media)),
        "recompensa_descontada_media": float(np.mean(rec_desc)),
    }


def imprime_resumo(nome, resumo):
    print(f"\n=== {nome} ===")
    print(f"  Episódios:              {resumo['n_episodios']}")
    print(f"  Taxa de sucesso:        {resumo['taxa_sucesso']:.1f}%")
    print(f"  Passos médios:          {resumo['passos_medios']:.1f} (+/- {resumo['desvio_passos']:.1f})")
    print(f"  Passos médios (sucesso):{resumo['passos_medios_sucesso']:.1f}")
    print(f"  Recompensa média:       {resumo['recompensa_media']:.2f}")
    print(f"  Recompensa descontada:  {resumo['recompensa_descontada_media']:.2f}")
