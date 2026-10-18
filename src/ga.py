"""(C) Algoritmo genético geracional com torneio, elitismo, cruzamento e mutação."""
from base import OrcamentoEsgotado


def algoritmo_genetico(prob, N=100, pc=0.9, pm=0.2, k_torneio=3, elite=2,
                       pares_diversidade=200):
    """Seleção por torneio: vence o indivíduo de menor h, então não é preciso
    converter custo em aptidão (o torneio só usa a ordem).

    Um filho idêntico ao pai (sem cruzamento nem mutação) herda o h do pai e
    não gasta avaliação. Para quando o orçamento de avaliações acaba.
    """
    rng = prob.rng
    geracoes = []  # (avaliações, melhor, média, distintos, dist. média)

    def torneio(pop, fit):
        idx = min(rng.sample(range(len(pop)), k_torneio), key=fit.__getitem__)
        return pop[idx], fit[idx]

    def registrar(pop, fit):
        pares = [rng.sample(range(len(pop)), 2) for _ in range(pares_diversidade)]
        dist = sum(prob.distancia(pop[a], pop[b]) for a, b in pares) / len(pares)
        geracoes.append((prob.avaliacoes, min(fit), sum(fit) / len(fit),
                         len(set(pop)), dist))

    try:
        pop = [prob.solucao_aleatoria() for _ in range(N)]
        fit = [prob.h(s) for s in pop]
        while True:
            registrar(pop, fit)
            ordem = sorted(range(N), key=fit.__getitem__)
            nova = [pop[i] for i in ordem[:elite]]
            nfit = [fit[i] for i in ordem[:elite]]
            while len(nova) < N:
                (p1, h1), (p2, h2) = torneio(pop, fit), torneio(pop, fit)
                if rng.random() < pc:
                    filhos = [[f, None] for f in prob.cruzamento(p1, p2)]
                else:
                    filhos = [[p1, h1], [p2, h2]]
                for f in filhos:
                    if rng.random() < pm:
                        f[0], f[1] = prob.mutacao(f[0]), None
                for f, hf in filhos[:N - len(nova)]:
                    nova.append(f)
                    nfit.append(prob.h(f) if hf is None else hf)
            pop, fit = nova, nfit
    except OrcamentoEsgotado:
        pass
    return {"melhor_h": prob.melhor_h, "melhor_s": prob.melhor_s,
            "geracoes": geracoes}
