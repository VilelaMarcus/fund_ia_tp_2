# TP2 — Busca Local e Algoritmo Genético

Fundamentos de Inteligência Artificial (UFMG). Este trabalho resolve o
sequenciamento de manutenção de ativos (atraso ponderado total) com três
algoritmos: greedy descent com random restart, simulated annealing e
algoritmo genético.

## Requisitos

- Python 3.11 com `numpy`, `pandas`, `scipy` e `matplotlib` (instale com `pip install -r requirements.txt`).
- LaTeX (`latexmk` + `pdflatex`), apenas para compilar o relatório.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `src/base.py` | Interface genérica `Problema` e contagem de avaliações (método `_contar`, o único ponto do código onde o orçamento é contado) |
| `src/manutencao.py` | O problema: leitura das instâncias, `h`, vizinhanças (swap, insertion, 2-opt), avaliação incremental, OX/PMX e mutação |
| `src/busca_local.py` | (A) Greedy descent com random restart: variantes best, first e sucessor aleatório |
| `src/sa.py` | (B) Simulated annealing com resfriamento geométrico e calibração de T0 |
| `src/ga.py` | (C) Algoritmo genético: torneio, elitismo, cruzamento e mutação |
| `src/heuristicas.py` | Guloso de referência e busca exata (usados só na verificação) |
| `src/experimentos.py` | Executa os experimentos e grava `resultados/*.csv` e `*_hist.json` |
| `src/analise.py` | Gera as tabelas (`Estudo_Dirigido_Fund_IA___2/tabelas/*.tex`) e as figuras (`Estudo_Dirigido_Fund_IA___2/figures/tp2/*.pdf`) |
| `tests/` | Testes automáticos |
| `Estudo_Dirigido_Fund_IA___2/` | Relatório em LaTeX; o arquivo principal é `relatorio.tex` |

## Como reproduzir

Todos os comandos são executados na raiz do repositório.

```bash
# 0. Testes: valores de verificação de h (4366, 3168, 1875 e o exemplo 39/22),
#    avaliação incremental = completa, OX/PMX válidos e parada exata no orçamento
python3 -m unittest discover -s tests -v

# 1. Experimentos (cerca de 12 min em 8 núcleos; use --proc N para limitar os processos)
python3 src/experimentos.py todos

# 2. Tabelas e figuras
python3 src/analise.py

# 3. Relatório
cd Estudo_Dirigido_Fund_IA___2 && latexmk -pdf relatorio.tex
```

Cada experimento também pode ser rodado sozinho. A tabela abaixo mostra o que cada um alimenta no relatório:

| Comando | Saída | Usado em |
|---|---|---|
| `python3 src/experimentos.py t1` | `resultados/t1.txt` | Seção 1 (saída da validação de h) |
| `python3 src/experimentos.py tempo` | `resultados/tempo.csv` | Tabela 1 (colunas de µs por sucessor) |
| `python3 src/experimentos.py t2` | `resultados/t2.csv` | Tabela 1 (comparação das vizinhanças) |
| `python3 src/experimentos.py calib_sa` | `resultados/calib_sa.csv` | Anexo A, Tabela 5 |
| `python3 src/experimentos.py calib_ga` | `resultados/calib_ga.csv` | Anexo A, Tabela 6 |
| `python3 src/experimentos.py t3` | `resultados/t3.csv`, `t3_hist.json` | Tabelas 2 e 3; Figuras 1 a 5 |
| `python3 src/experimentos.py extra` | `resultados/extra.csv` | Tabela 4 (experimento adicional do GA) |

Depois de qualquer um deles, rode `python3 src/analise.py` para regenerar as
tabelas e figuras. Os números citados no texto também ficam em
`resultados/resumo.json`.

## Reprodutibilidade

- Cada execução usa uma semente fixa: `0..29` no protocolo principal, `500..` no T2, `1000..` e `2000..` nas calibrações.
- Os tempos de parede variam de máquina para máquina e foram medidos com 8 processos em paralelo. Já os valores de h são determinísticos.
- Os parâmetros escolhidos nas calibrações estão em `PARAM_SA` e `PARAM_GA` (`src/experimentos.py`), e a vizinhança adotada está em `VIZ_ADOTADA`.
