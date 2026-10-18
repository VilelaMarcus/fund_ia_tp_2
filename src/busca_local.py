"""(A) Greedy descent com random restart. Genérico: só usa a interface Problema."""
from base import OrcamentoEsgotado

VARIANTES = ("best", "first", "aleatorio")


def descida_gulosa(prob, variante="best", max_falhas=None):
    """Desce até um ótimo local e reinicia de uma solução aleatória até
    esgotar o orçamento.

    best      — varre toda a vizinhança e move para o melhor sucessor;
    first     — varre em ordem aleatória e aceita o primeiro que melhora;
    aleatorio — sorteia um sucessor e o aceita se melhorar; declara ótimo
                local após ``max_falhas`` sorteios seguidos sem melhora
                (padrão: o tamanho da vizinhança).
    """
    if max_falhas is None:
        max_falhas = prob.tamanho_vizinhanca()
    passos = 0          # movimentos de melhoria aceitos
    otimos = []         # (h, s) de cada ótimo local completo
    try:
        while True:
            s = prob.solucao_aleatoria()
            hs = prob.h(s)
            while True:
                if variante == "best":
                    melhor_mov, melhor_v = None, hs
                    for mov, v in prob.sucessores(s, limite=melhor_v):
                        if v < melhor_v:
                            melhor_mov, melhor_v = mov, v
                elif variante == "first":
                    melhor_mov = None
                    for mov, v in prob.sucessores(s, limite=hs, embaralhar=True):
                        if v < hs:
                            melhor_mov, melhor_v = mov, v
                            break
                else:
                    melhor_mov, falhas = None, 0
                    while falhas < max_falhas:
                        mov, v = prob.sucessor_aleatorio(s, limite=hs)
                        if v < hs:
                            melhor_mov, melhor_v = mov, v
                            break
                        falhas += 1
                if melhor_mov is None:
                    otimos.append((hs, s))
                    break
                s, hs = prob.aplicar(s, melhor_mov), melhor_v
                passos += 1
    except OrcamentoEsgotado:
        pass
    return {"melhor_h": prob.melhor_h, "melhor_s": prob.melhor_s,
            "passos": passos, "reinicios": len(otimos), "otimos": otimos}
