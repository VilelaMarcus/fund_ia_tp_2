"""Interface genérica entre algoritmos e problemas.

Os algoritmos (busca local, simulated annealing, algoritmo genético) só
conhecem os métodos desta classe. Toda avaliação de h passa por
``_contar``, o único ponto do código onde o orçamento é contabilizado.
"""
import math


class OrcamentoEsgotado(Exception):
    """Lançada quando se tenta avaliar h além do orçamento."""


class Problema:
    """Classe base de um problema de minimização.

    Subclasses implementam:
        solucao_aleatoria() -> s
        h(s) -> float                         (conta 1 avaliação)
        sucessores(s, limite) -> iter[(mov, h)] (cada sucessor conta 1)
        sucessor_aleatorio(s, limite) -> (mov, h)  (conta 1)
        aplicar(s, mov) -> s'
        tamanho_vizinhanca() -> int
        cruzamento(s1, s2) -> (f1, f2)
        mutacao(s) -> s'
        distancia(s1, s2) -> float

    ``limite``: quando o valor de um sucessor é >= limite, o problema pode
    devolver qualquer número >= limite (avaliação interrompida). Valores
    menores que o limite são sempre exatos.
    """

    def __init__(self, orcamento=math.inf):
        self.reiniciar(orcamento)

    def reiniciar(self, orcamento=None):
        if orcamento is not None:
            self.orcamento = orcamento
        self.avaliacoes = 0
        self.melhor_h = math.inf
        self.melhor_s = None
        # (avaliações gastas, melhor h até ali): curva de convergência
        self.historico = []

    def _contar(self):
        if self.avaliacoes >= self.orcamento:
            raise OrcamentoEsgotado
        self.avaliacoes += 1

    def _registrar(self, valor, construir):
        """Atualiza o melhor global; ``construir`` só é chamado se melhorou."""
        if valor < self.melhor_h:
            self.melhor_h = valor
            self.melhor_s = construir()
            self.historico.append((self.avaliacoes, valor))
