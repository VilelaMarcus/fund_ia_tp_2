"""Heurística gulosa e busca exata, usadas apenas para verificação (T1)."""
from manutencao import avaliar


def guloso_menor_atraso(inst):
    """Escolhe sempre o ativo de menor acréscimo imediato de atraso.
    Empate: o mais próximo; depois o menor índice (reproduz h = 3168)."""
    T, p, d, w = inst.T, inst.p, inst.d, inst.w
    restantes = set(range(1, inst.n + 1))
    s, C, prev = [], 0, 0
    while restantes:
        def chave(j):
            c = C + T[prev][j] + p[j]
            return (w[j] * max(0, c - d[j]), T[prev][j], j)
        j = min(restantes, key=chave)
        C += T[prev][j] + p[j]
        s.append(j)
        restantes.remove(j)
        prev = j
    return tuple(s), avaliar(inst, s)


def otimo_exato(inst):
    """Programação dinâmica sobre (conjunto visitado, último ativo) guardando
    a fronteira de Pareto de (tempo atual, custo acumulado): um rótulo com
    tempo e custo maiores ou iguais a outro nunca leva a solução melhor.
    Viável para n pequeno (n = 12 -> 4096 subconjuntos)."""
    n, T, p, d, w = inst.n, inst.T, inst.p, inst.d, inst.w
    # rótulos[(mask, ultimo)] = lista de (C, custo, sequência)
    camada = {}
    for j in range(1, n + 1):
        C = T[0][j] + p[j]
        camada[(1 << (j - 1), j)] = [(C, w[j] * max(0, C - d[j]), (j,))]
    for _ in range(n - 1):
        nova = {}
        for (mask, u), rotulos in camada.items():
            for j in range(1, n + 1):
                bit = 1 << (j - 1)
                if mask & bit:
                    continue
                chave = (mask | bit, j)
                lst = nova.setdefault(chave, [])
                for C, custo, seq in rotulos:
                    C2 = C + T[u][j] + p[j]
                    lst.append((C2, custo + w[j] * max(0, C2 - d[j]), seq + (j,)))
        for chave, lst in nova.items():
            lst.sort()
            fronteira, melhor = [], float("inf")
            for r in lst:
                if r[1] < melhor:
                    fronteira.append(r)
                    melhor = r[1]
            nova[chave] = fronteira
        camada = nova
    _, custo, seq = min((r for lst in camada.values() for r in lst),
                        key=lambda r: r[1])
    return seq, custo
