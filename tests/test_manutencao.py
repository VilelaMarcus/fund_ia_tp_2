import math
import os
import random
import sys
import unittest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

from base import OrcamentoEsgotado  # noqa: E402
from heuristicas import guloso_menor_atraso, otimo_exato  # noqa: E402
from manutencao import (VIZINHANCAS, Instancia, ProblemaManutencao,  # noqa: E402
                        avaliar, movimentos, order_crossover, pmx)


def instancia(nome):
    return Instancia.ler(os.path.join(RAIZ, "data", f"manutencao_{nome}.txt"))


class TestFitness(unittest.TestCase):
    def test_identidade_pequena(self):
        self.assertEqual(avaliar(instancia("pequena"), range(1, 13)), 4366)

    def test_guloso_pequena(self):
        self.assertEqual(guloso_menor_atraso(instancia("pequena"))[1], 3168)

    def test_otimo_exato_pequena(self):
        self.assertEqual(otimo_exato(instancia("pequena"))[1], 1875)

    def test_exemplo_figura1(self):
        # Deslocamentos lidos da Figura 1 (os pares não usados ficam com 99).
        T = [[99] * 5 for _ in range(5)]
        for (i, j), t in {(0, 1): 5, (1, 4): 5, (4, 2): 8, (2, 3): 10,
                          (4, 3): 6}.items():
            T[i][j] = T[j][i] = t
        inst = Instancia(4, T, [0, 2, 3, 1, 4], [0, 10, 14, 25, 20],
                         [0, 3, 1, 2, 5])
        self.assertEqual(avaliar(inst, (1, 4, 2, 3)), 39)
        self.assertEqual(avaliar(inst, (1, 4, 3, 2)), 22)


class TestVizinhancas(unittest.TestCase):
    def test_exemplos_figura3(self):
        s = (3, 7, 1, 5, 2, 8)
        esperado = {"swap": (3, 2, 1, 5, 7, 8), "insertion": (3, 1, 5, 2, 7, 8),
                    "2opt": (3, 2, 5, 1, 7, 8)}
        for viz, alvo in esperado.items():
            prob = ProblemaManutencao(instancia("pequena"), vizinhanca=viz)
            self.assertEqual(prob.aplicar(s, (1, 4)), alvo)

    def test_tamanho_e_sem_duplicatas(self):
        n = 12
        s = tuple(range(1, n + 1))
        tamanhos = {"swap": n * (n - 1) // 2, "insertion": (n - 1) ** 2,
                    "2opt": n * (n - 1) // 2}
        for viz in VIZINHANCAS:
            prob = ProblemaManutencao(instancia("pequena"), vizinhanca=viz)
            vizinhos = {prob.aplicar(s, m) for m in movimentos(n, viz)}
            self.assertEqual(len(vizinhos), tamanhos[viz])
            self.assertNotIn(s, vizinhos)

    def test_incremental_igual_completa(self):
        rng = random.Random(0)
        for nome in ("pequena", "media", "grande"):
            inst = instancia(nome)
            for viz in VIZINHANCAS:
                prob = ProblemaManutencao(inst, vizinhanca=viz, semente=1)
                for _ in range(5):
                    s = prob.solucao_aleatoria()
                    for mov in rng.sample(prob.movs, min(200, len(prob.movs))):
                        exato = avaliar(inst, prob.aplicar(s, mov))
                        self.assertEqual(prob._avaliar_mov(s, mov, math.inf), exato)
                        # com limite: exato se abaixo, senão algo >= limite
                        lim = exato + rng.choice([-50, 0, 1, 50])
                        v = prob._avaliar_mov(s, mov, lim)
                        if exato < lim:
                            self.assertEqual(v, exato)
                        else:
                            self.assertGreaterEqual(v, lim)


class TestOperadoresGeneticos(unittest.TestCase):
    def test_ox_exemplo(self):
        p1 = (1, 2, 3, 4, 5, 6, 7, 8, 9)
        p2 = (9, 3, 7, 8, 2, 6, 5, 1, 4)
        self.assertEqual(order_crossover(p1, p2, 3, 6), (3, 8, 2, 4, 5, 6, 7, 1, 9))

    def test_pmx_exemplo(self):
        p1 = (1, 2, 3, 4, 5, 6, 7, 8, 9)
        p2 = (9, 3, 7, 8, 2, 6, 5, 1, 4)
        self.assertEqual(pmx(p1, p2, 3, 6), (9, 3, 2, 4, 5, 6, 7, 1, 8))

    def test_filhos_sao_permutacoes(self):
        for op in ("ox", "pmx"):
            prob = ProblemaManutencao(instancia("media"), op_cruzamento=op, semente=3)
            for _ in range(300):
                f1, f2 = prob.cruzamento(prob.solucao_aleatoria(),
                                         prob.solucao_aleatoria())
                for f in (f1, f2, prob.mutacao(f1)):
                    self.assertEqual(sorted(f), list(range(1, 51)))


class TestOrcamento(unittest.TestCase):
    def test_para_exatamente(self):
        prob = ProblemaManutencao(instancia("pequena"), orcamento=100, semente=0)
        s = prob.solucao_aleatoria()
        prob.h(s)
        with self.assertRaises(OrcamentoEsgotado):
            for _ in prob.sucessores(s):
                pass
        self.assertEqual(prob.avaliacoes, 100)
        self.assertEqual(prob.melhor_h, min(v for _, v in prob.historico))


if __name__ == "__main__":
    unittest.main()
