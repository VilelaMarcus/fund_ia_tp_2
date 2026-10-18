"""Sequenciamento de manutenção de ativos (atraso ponderado total).

Uma solução é uma tupla com a permutação dos ativos 1..n (ordem de visita).
"""
import itertools
import math
import random

from base import Problema

VIZINHANCAS = ("swap", "insertion", "2opt")


class Instancia:
    """Dados de uma instância. Índice 0 é a base (p0 = 0)."""

    def __init__(self, n, T, p, d, w, nome=""):
        self.n, self.T, self.p, self.d, self.w, self.nome = n, T, p, d, w, nome

    @classmethod
    def ler(cls, caminho, nome=None):
        with open(caminho) as f:
            v = f.read().split()
        n = int(v[0])
        pts = [(int(v[1]), int(v[2]))]
        p, d, w = [0], [0], [0]
        for j in range(n):
            x, y, pj, dj, wj = map(int, v[3 + 5 * j: 8 + 5 * j])
            pts.append((x, y))
            p.append(pj)
            d.append(dj)
            w.append(wj)
        # Equação (1): deslocamento arredondado para o inteiro mais próximo.
        # sqrt de inteiro nunca termina em ,5 exato, então round() não empata.
        T = [[round(math.dist(a, b)) for b in pts] for a in pts]
        return cls(n, T, p, d, w, nome or caminho)


def avaliar(inst, s):
    """Equações (2)-(3), avaliação completa O(n). Não conta orçamento."""
    T, p, d, w = inst.T, inst.p, inst.d, inst.w
    C = custo = prev = 0
    for j in s:
        C += T[prev][j] + p[j]
        if C > d[j]:
            custo += w[j] * (C - d[j])
        prev = j
    return custo


def cronograma(inst, s):
    """Tempos de conclusão C[k] e custo acumulado F[k] por posição."""
    T, p, d, w = inst.T, inst.p, inst.d, inst.w
    C, F = [], []
    c = custo = prev = 0
    for j in s:
        c += T[prev][j] + p[j]
        if c > d[j]:
            custo += w[j] * (c - d[j])
        C.append(c)
        F.append(custo)
        prev = j
    return C, F


def segmento(s, viz, i, j):
    """Trecho alterado por um movimento: (a, b, novo s[a..b])."""
    if viz == "swap":
        return i, j, (s[j],) + s[i + 1:j] + (s[i],)
    if viz == "2opt":
        return i, j, s[i:j + 1][::-1]
    # insertion: remove o ativo da posição i e o reinsere na posição j
    if i < j:
        return i, j, s[i + 1:j + 1] + (s[i],)
    return j, i, (s[i],) + s[j:i]


def movimentos(n, viz):
    if viz in ("swap", "2opt"):
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    # mover i -> i-1 é idêntico a mover i-1 -> i: descarta a duplicata
    return [(i, j) for i in range(n) for j in range(n) if j != i and j != i - 1]


class ProblemaManutencao(Problema):
    def __init__(self, inst, vizinhanca="insertion", op_mutacao="swap",
                 op_cruzamento="ox", incremental=True, semente=None,
                 orcamento=math.inf):
        super().__init__(orcamento)
        assert vizinhanca in VIZINHANCAS and op_mutacao in VIZINHANCAS
        self.inst = inst
        self.viz = vizinhanca
        self.op_mutacao = op_mutacao
        self.op_cruzamento = op_cruzamento
        self.incremental = incremental
        self.rng = random.Random(semente)
        self.movs = movimentos(inst.n, vizinhanca)
        self._movs_embaralhados = None
        self._cache_s = None  # solução cujo cronograma está em _C/_F

    # ------------------------------------------------------------------ básico
    def solucao_aleatoria(self):
        s = list(range(1, self.inst.n + 1))
        self.rng.shuffle(s)
        return tuple(s)

    def h(self, s):
        self._contar()
        v = avaliar(self.inst, s)
        self._registrar(v, lambda: s)
        return v

    def tamanho_vizinhanca(self):
        return len(self.movs)

    def aplicar(self, s, mov):
        a, b, seg = segmento(s, self.viz, *mov)
        return s[:a] + seg + s[b + 1:]

    # -------------------------------------------------------------- vizinhança
    def _preparar(self, s):
        if self._cache_s is not s:
            self._C, self._F = cronograma(self.inst, s)
            d, w = self.inst.d, self.inst.w
            self._folga = [d[j] - c for j, c in zip(s, self._C)]
            self._W = [w[j] for j in s]
            self._cache_s = s

    def _avaliar_mov(self, s, mov, limite):
        """Avaliação incremental de um movimento sobre s.

        O prefixo s[0..a-1] não muda: seu custo vem do cronograma de s.
        O trecho alterado s[a..b] é recalculado. Da posição b+1 em diante os
        ativos são os mesmos e o atraso de tempo delta é constante, então
            custo_k = w_k * max(0, delta - folga_k),  folga_k = d_k - C_k(s);
        se delta = 0 o sufixo custa o mesmo que em s. Como h é não
        decrescente no prefixo, a soma é interrompida ao atingir ``limite``.
        """
        if not self.incremental:
            return avaliar(self.inst, self.aplicar(s, mov))
        self._preparar(s)
        Cv, Fv = self._C, self._F
        T, p, d, w = self.inst.T, self.inst.p, self.inst.d, self.inst.w
        a, b, seg = segmento(s, self.viz, *mov)
        if a == 0:
            prev = C = custo = 0
        else:
            prev, C, custo = s[a - 1], Cv[a - 1], Fv[a - 1]
        for j in seg:
            C += T[prev][j] + p[j]
            if C > d[j]:
                custo += w[j] * (C - d[j])
            prev = j
        n = len(s)
        b += 1
        if b == n:
            return custo
        j = s[b]
        delta = C + T[prev][j] + p[j] - Cv[b]
        if delta == 0:
            return custo + Fv[n - 1] - Fv[b - 1]
        if custo >= limite:
            return custo
        folga, W = self._folga, self._W
        for k in range(b, n):
            x = delta - folga[k]
            if x > 0:
                custo += W[k] * x
                if custo >= limite:
                    return custo
        return custo

    def sucessores(self, s, limite=math.inf, embaralhar=False):
        """embaralhar: varre a vizinhança numa ordem aleatória (lista
        embaralhada uma vez, percorrida circularmente a partir de um ponto
        sorteado), sem o custo O(|vizinhança|) de embaralhar a cada passo."""
        movs = self.movs
        if embaralhar:
            if self._movs_embaralhados is None:
                self._movs_embaralhados = movs[:]
                self.rng.shuffle(self._movs_embaralhados)
            movs = self._movs_embaralhados
            ini = self.rng.randrange(len(movs))
            movs = itertools.chain(itertools.islice(movs, ini, None),
                                   itertools.islice(movs, ini))
        for mov in movs:
            self._contar()
            v = self._avaliar_mov(s, mov, limite)
            self._registrar(v, lambda: self.aplicar(s, mov))
            yield mov, v

    def sucessor_aleatorio(self, s, limite=math.inf):
        mov = self.movs[self.rng.randrange(len(self.movs))]
        self._contar()
        v = self._avaliar_mov(s, mov, limite)
        self._registrar(v, lambda: self.aplicar(s, mov))
        return mov, v

    # ------------------------------------------------------------- genéticos
    def cruzamento(self, s1, s2):
        n = len(s1)
        a, b = sorted(self.rng.sample(range(n), 2))
        op = order_crossover if self.op_cruzamento == "ox" else pmx
        return op(s1, s2, a, b), op(s2, s1, a, b)

    def mutacao(self, s):
        n = len(s)
        i, j = self.rng.sample(range(n), 2)
        if self.op_mutacao != "insertion" and i > j:
            i, j = j, i
        a, b, seg = segmento(s, self.op_mutacao, i, j)
        return s[:a] + seg + s[b + 1:]

    def distancia(self, s1, s2):
        """Distância de Hamming posicional normalizada (0 = iguais)."""
        return sum(x != y for x, y in zip(s1, s2)) / len(s1)


def order_crossover(p1, p2, a, b):
    """OX: o filho herda p1[a..b] na mesma posição; as posições restantes são
    preenchidas, a partir de b+1 (circular), com os genes de p2 na ordem em
    que aparecem a partir de b+1, pulando os já presentes."""
    n = len(p1)
    filho = [None] * n
    filho[a:b + 1] = p1[a:b + 1]
    presentes = set(p1[a:b + 1])
    ordem = [p2[(b + 1 + k) % n] for k in range(n)]
    pos = (b + 1) % n
    for g in ordem:
        if g not in presentes:
            filho[pos] = g
            pos = (pos + 1) % n
    return tuple(filho)


def pmx(p1, p2, a, b):
    """PMX: o filho herda p1[a..b]; os genes de p2 em conflito são realocados
    seguindo o mapeamento p1[k] <-> p2[k] do trecho central."""
    n = len(p1)
    filho = [None] * n
    filho[a:b + 1] = p1[a:b + 1]
    pos_em_p2 = {g: k for k, g in enumerate(p2)}
    presentes = set(p1[a:b + 1])
    for k in range(a, b + 1):
        g = p2[k]
        if g in presentes:
            continue
        alvo = k
        while a <= alvo <= b:
            alvo = pos_em_p2[p1[alvo]]
        filho[alvo] = g
    for k in range(n):
        if filho[k] is None:
            filho[k] = p2[k]
    return tuple(filho)
