"""(B) Simulated annealing com resfriamento geométrico T <- alpha*T."""
import math

from base import OrcamentoEsgotado


def calibrar_t0(prob, s, hs, amostras=200, chi0=0.8):
    """T0 tal que a piora média observada em ``amostras`` sucessores
    aleatórios de s seja aceita com probabilidade chi0:
        exp(-media(delta+) / T0) = chi0.
    As avaliações gastas aqui contam no orçamento."""
    deltas = []
    for _ in range(amostras):
        _, v = prob.sucessor_aleatorio(s)
        if v > hs:
            deltas.append(v - hs)
    media = sum(deltas) / len(deltas) if deltas else 1.0
    return -media / math.log(chi0)


def simulated_annealing(prob, chi0=0.8, tf=0.5, amostras_calib=200,
                        janela=None):
    """T0 é calibrado na solução inicial (ver ``calibrar_t0``). Tf é fixo:
    com tf = 0,5 uma piora de 1 (a menor possível, pois h é inteiro) é
    aceita com prob. e^-2 ~ 13% e uma piora de 5 com ~0,005%, ou seja, o
    final é praticamente greedy descent. alpha é escolhido para que T caia de
    T0 a Tf exatamente no orçamento restante, um passo por avaliação.

    Registra, a cada ``janela`` avaliações, a taxa de aceitação de pioras.
    """
    orcamento = prob.orcamento
    janela = janela or max(1, orcamento // 200)
    serie = []  # (avaliações, T, taxa de aceitação de pioras)
    T = t0 = None
    try:
        s = prob.solucao_aleatoria()
        hs = prob.h(s)
        t0 = calibrar_t0(prob, s, hs, amostras_calib, chi0)
        restante = orcamento - prob.avaliacoes
        alpha = (tf / t0) ** (1.0 / restante) if t0 > tf else 1.0
        T = t0
        piores = aceitas = 0
        while True:
            # limite: pioras com prob. < 1e-12 nem precisam de valor exato
            mov, v = prob.sucessor_aleatorio(s, limite=hs + 28 * T)
            if v <= hs:
                s, hs = prob.aplicar(s, mov), v
            else:
                piores += 1
                if v < hs + 28 * T and prob.rng.random() < math.exp(-(v - hs) / T):
                    s, hs = prob.aplicar(s, mov), v
                    aceitas += 1
            T *= alpha
            if prob.avaliacoes % janela == 0:
                serie.append((prob.avaliacoes, T, aceitas / piores if piores else 0.0))
                piores = aceitas = 0
    except OrcamentoEsgotado:
        pass
    return {"melhor_h": prob.melhor_h, "melhor_s": prob.melhor_s,
            "t0": t0, "tf": tf, "serie_aceitacao": serie}
