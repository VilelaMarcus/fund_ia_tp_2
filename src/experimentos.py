"""Executa os experimentos do TP2 e grava tudo em resultados/.

Uso (a partir da raiz do repositório):
    python3 src/experimentos.py t1          # verificação da função h
    python3 src/experimentos.py tempo       # custo por movimento (incremental x completa)
    python3 src/experimentos.py t2          # comparação das vizinhanças
    python3 src/experimentos.py calib_sa    # calibração do simulated annealing
    python3 src/experimentos.py calib_ga    # calibração do algoritmo genético
    python3 src/experimentos.py t3          # protocolo principal (Seção 3)
    python3 src/experimentos.py extra       # experimento adicional do GA
    python3 src/experimentos.py todos       # tudo acima, em ordem

Cada execução recebe uma semente própria, então os resultados são
reproduzíveis. Para rodar em paralelo use --proc (padrão: nº de CPUs).
"""
import argparse
import csv
import itertools
import json
import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from busca_local import descida_gulosa  # noqa: E402
from ga import algoritmo_genetico  # noqa: E402
from heuristicas import guloso_menor_atraso, otimo_exato  # noqa: E402
from manutencao import (VIZINHANCAS, Instancia, ProblemaManutencao,  # noqa: E402
                        avaliar)
from sa import simulated_annealing  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTADOS = os.path.join(RAIZ, "resultados")

INSTANCIAS = ("pequena", "media", "grande")
ORCAMENTO = {"pequena": 50_000, "media": 150_000, "grande": 300_000}
EXECUCOES = {"pequena": 30, "media": 30, "grande": 10}
REFERENCIA = {"pequena": 1875, "media": 16332, "grande": 60417}

# Vizinhança adotada após o T2 (ver relatório, Seção de vizinhanças).
VIZ_ADOTADA = "insertion"

# Parâmetros escolhidos nas calibrações (calib_sa / calib_ga).
PARAM_SA = {"pequena": {"chi0": 0.5, "tf": 1.0},
            "media": {"chi0": 0.8, "tf": 1.0},
            "grande": {"chi0": 0.2, "tf": 0.2}}
PARAM_GA = {i: {"N": 50, "pm": 0.5, "op_mutacao": "insertion"}
            for i in ("pequena", "media", "grande")}

_cache_inst = {}


def carregar(nome):
    if nome not in _cache_inst:
        _cache_inst[nome] = Instancia.ler(
            os.path.join(RAIZ, "data", f"manutencao_{nome}.txt"), nome)
    return _cache_inst[nome]


# --------------------------------------------------------------- uma execução
def executar(cfg):
    """cfg: dict com exp, alg, inst, semente, viz e parâmetros do algoritmo."""
    inst = carregar(cfg["inst"])
    par = dict(cfg.get("par", {}))
    prob = ProblemaManutencao(
        inst, vizinhanca=cfg.get("viz", VIZ_ADOTADA),
        op_mutacao=par.pop("op_mutacao", "swap"),
        op_cruzamento=par.pop("op_cruzamento", "ox"),
        semente=cfg["semente"], orcamento=ORCAMENTO[cfg["inst"]])
    t = time.perf_counter()
    alg = cfg["alg"]
    if alg.startswith("gd_"):
        r = descida_gulosa(prob, variante=alg[3:], **par)
    elif alg == "sa":
        r = simulated_annealing(prob, **par)
    elif alg == "ga":
        r = algoritmo_genetico(prob, **par)
    else:
        raise ValueError(alg)
    tempo = time.perf_counter() - t
    assert avaliar(inst, r["melhor_s"]) == r["melhor_h"]
    linha = {k: cfg[k] for k in ("exp", "alg", "inst", "semente")}
    linha.update(viz=cfg.get("viz", VIZ_ADOTADA), rotulo=cfg.get("rotulo", alg),
                 par=json.dumps(cfg.get("par", {}), sort_keys=True),
                 melhor_h=r["melhor_h"], avaliacoes=prob.avaliacoes,
                 tempo_s=round(tempo, 4))
    hist = {"id": cfg["id"], "historico": prob.historico,
            "melhor_s": list(r["melhor_s"])}
    if alg.startswith("gd_"):
        distintos = {s for _, s in r["otimos"]}
        linha.update(passos=r["passos"], reinicios=r["reinicios"],
                     otimos_distintos=len(distintos),
                     otimos_globais=sum(h == REFERENCIA["pequena"]
                                        for h, _ in r["otimos"])
                     if cfg["inst"] == "pequena" else "")
        if cfg["inst"] == "pequena":
            hist["otimos"] = [[h, list(s)] for h, s in r["otimos"]]
    elif alg == "sa":
        linha.update(t0=round(r["t0"], 3), tf=r["tf"])
        hist["serie_aceitacao"] = r["serie_aceitacao"]
    elif alg == "ga":
        hist["geracoes"] = r["geracoes"]
        linha.update(geracoes=len(r["geracoes"]))
    return linha, hist


def rodar(nome_exp, cfgs, proc):
    for k, c in enumerate(cfgs):
        c["exp"], c["id"] = nome_exp, f"{nome_exp}-{k}"
    t = time.time()
    with Pool(proc) as pool:
        saidas = []
        for k, saida in enumerate(pool.imap_unordered(executar, cfgs), 1):
            saidas.append(saida)
            print(f"\r{nome_exp}: {k}/{len(cfgs)} ({time.time() - t:.0f}s)",
                  end="", flush=True)
    print()
    saidas.sort(key=lambda o: int(o[1]["id"].rsplit("-", 1)[1]))
    linhas = [o[0] for o in saidas]
    campos = list(dict.fromkeys(k for l in linhas for k in l))
    os.makedirs(RESULTADOS, exist_ok=True)
    with open(os.path.join(RESULTADOS, f"{nome_exp}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, campos)
        wr.writeheader()
        wr.writerows(linhas)
    with open(os.path.join(RESULTADOS, f"{nome_exp}_hist.json"), "w") as f:
        json.dump({o[1]["id"]: o[1] for o in saidas}, f)
    return linhas


def sementes(inst, n=None, base=0):
    return range(base, base + (n or EXECUCOES[inst]))


# ---------------------------------------------------------------- experimentos
def exp_t1(_proc):
    linhas = []
    for nome in INSTANCIAS:
        inst = carregar(nome)
        linhas.append(f"{nome}: h(identidade) = {avaliar(inst, range(1, inst.n + 1))}")
        s, h = guloso_menor_atraso(inst)
        linhas.append(f"{nome}: guloso (menor acréscimo de atraso) = {h}")
    s, h = otimo_exato(carregar("pequena"))
    linhas.append(f"pequena: ótimo exato (PD com fronteira de Pareto) = {h}, "
                  f"pi* = {s}")
    texto = "\n".join(linhas)
    print(texto)
    os.makedirs(RESULTADOS, exist_ok=True)
    with open(os.path.join(RESULTADOS, "t1.txt"), "w") as f:
        f.write(texto + "\n")


def exp_tempo(_proc, movs=20_000, orc_descida=30_000):
    """Tempo médio por avaliação de sucessor, incremental x completa, em dois
    cenários: (i) movimentos aleatórios sem limite (só o ganho do sufixo);
    (ii) dentro de uma descida first improvement, onde o limite h(s) também
    interrompe a avaliação. Em (ii) a trajetória é idêntica nos dois modos."""
    linhas = []
    for nome, viz in itertools.product(INSTANCIAS, VIZINHANCAS):
        inst = carregar(nome)
        res = {}
        for inc in (True, False):
            prob = ProblemaManutencao(inst, vizinhanca=viz, incremental=inc,
                                      semente=7)
            rng = random.Random(7)
            sols = [prob.solucao_aleatoria() for _ in range(20)]
            t = time.perf_counter()
            for k in range(movs):  # movs/20 sucessores de cada solução
                prob._avaliar_mov(sols[k * 20 // movs], rng.choice(prob.movs),
                                  float("inf"))
            res["aleat", inc] = (time.perf_counter() - t) / movs * 1e6
            prob = ProblemaManutencao(inst, vizinhanca=viz, incremental=inc,
                                      semente=7, orcamento=orc_descida)
            t = time.perf_counter()
            r = descida_gulosa(prob, "first")
            res["desc", inc] = (time.perf_counter() - t) / orc_descida * 1e6
            res["h", inc] = r["melhor_h"]
        assert res["h", True] == res["h", False]
        linhas.append({"inst": nome, "viz": viz, "n": inst.n,
                       "tam_vizinhanca": len(prob.movs),
                       "us_aleat_incremental": round(res["aleat", True], 2),
                       "us_aleat_completa": round(res["aleat", False], 2),
                       "ganho_aleat": round(res["aleat", False] / res["aleat", True], 2),
                       "us_descida_incremental": round(res["desc", True], 2),
                       "us_descida_completa": round(res["desc", False], 2),
                       "ganho_descida": round(res["desc", False] / res["desc", True], 2)})
        print(linhas[-1])
    with open(os.path.join(RESULTADOS, "tempo.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, list(linhas[0]))
        wr.writeheader()
        wr.writerows(linhas)


def exp_t2(proc):
    cfgs = [{"alg": f"gd_{v}", "inst": i, "viz": viz, "semente": s}
            for i in INSTANCIAS for viz in VIZINHANCAS for v in ("best", "first")
            for s in sementes(i, 10, base=500)]
    rodar("t2", cfgs, proc)


def exp_calib_sa(proc):
    cfgs = [{"alg": "sa", "inst": i, "semente": s,
             "par": {"chi0": c, "tf": tf}, "rotulo": f"chi0={c} tf={tf}"}
            for i in INSTANCIAS for c in (0.2, 0.5, 0.8) for tf in (0.2, 1.0, 5.0)
            for s in sementes(i, 5, base=1000)]
    rodar("calib_sa", cfgs, proc)


def exp_calib_ga(proc):
    cfgs = [{"alg": "ga", "inst": i, "semente": s,
             "par": {"N": N, "pm": pm, "op_mutacao": op},
             "rotulo": f"N={N} pm={pm} mut={op}"}
            for i in INSTANCIAS for N in (50, 100, 200) for pm in (0.2, 0.5, 1.0)
            for op in ("swap", "insertion") for s in sementes(i, 5, base=2000)]
    rodar("calib_ga", cfgs, proc)


def exp_t3(proc):
    cfgs = []
    for i in INSTANCIAS:
        for s in sementes(i):
            for v in ("best", "first", "aleatorio"):
                cfgs.append({"alg": f"gd_{v}", "inst": i, "semente": s})
            cfgs.append({"alg": "sa", "inst": i, "semente": s, "par": PARAM_SA[i]})
            cfgs.append({"alg": "ga", "inst": i, "semente": s, "par": PARAM_GA[i]})
    rodar("t3", cfgs, proc)


def exp_extra(proc):
    """Hipótese: a recombinação (OX/PMX) não combina bons trechos dos pais.
    Compara o GA com OX, com PMX e sem cruzamento (pc = 0, só mutação)."""
    cfgs = []
    for i in INSTANCIAS:
        for s in sementes(i):
            for rot, extra in (("ox", {"op_cruzamento": "ox"}),
                               ("pmx", {"op_cruzamento": "pmx"}),
                               ("sem_cruzamento", {"pc": 0.0})):
                cfgs.append({"alg": "ga", "inst": i, "semente": s, "rotulo": rot,
                             "par": {**PARAM_GA[i], **extra}})
    rodar("extra", cfgs, proc)


EXPERIMENTOS = {"t1": exp_t1, "tempo": exp_tempo, "t2": exp_t2,
                "calib_sa": exp_calib_sa, "calib_ga": exp_calib_ga,
                "t3": exp_t3, "extra": exp_extra}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=[*EXPERIMENTOS, "todos"])
    ap.add_argument("--proc", type=int, default=os.cpu_count())
    a = ap.parse_args()
    for nome in (EXPERIMENTOS if a.exp == "todos" else [a.exp]):
        EXPERIMENTOS[nome](a.proc)
