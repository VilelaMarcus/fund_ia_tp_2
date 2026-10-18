"""Gera as tabelas (.tex) e figuras (.pdf) do relatório a partir de resultados/.

Uso: python3 src/analise.py      (depois de rodar src/experimentos.py todos)
Saída: Estudo_Dirigido_Fund_IA___2/tabelas/*.tex e figures/tp2/*.pdf e
       resultados/resumo.json (números citados no texto).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(RAIZ, "resultados")
TAB = os.path.join(RAIZ, "Estudo_Dirigido_Fund_IA___2", "tabelas")
FIG = os.path.join(RAIZ, "Estudo_Dirigido_Fund_IA___2", "figures", "tp2")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

INST = ["pequena", "media", "grande"]
NOME_INST = {"pequena": "Pequena ($n=12$)", "media": "Intermediária ($n=50$)",
             "grande": "Grande ($n=150$)"}
TITULO_INST = {"pequena": "Pequena (n = 12)", "media": "Intermediária (n = 50)",
               "grande": "Grande (n = 150)"}
ORCAMENTO = {"pequena": 50_000, "media": 150_000, "grande": 300_000}
H_ESTRELA = 1875
REFERENCIA = {"pequena": 1875, "media": 16332, "grande": 60417}

# Ordem fixa e cor fixa por algoritmo em todas as figuras.
ALGS = ["gd_best", "gd_first", "gd_aleatorio", "sa", "ga"]
NOME_ALG = {"gd_best": "GD best", "gd_first": "GD first",
            "gd_aleatorio": "GD aleatório", "sa": "SA", "ga": "GA"}
COR = {"gd_first": "#2a78d6", "gd_aleatorio": "#eb6834", "sa": "#1baf7a",
       "ga": "#6250d6", "gd_best": "#52514e"}
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": TINTA2,
    "axes.labelcolor": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRADE, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "legend.frameon": False, "savefig.bbox": "tight"})


def ler(exp, historico=False):
    df = pd.read_csv(os.path.join(RES, f"{exp}.csv"))
    if not historico:
        return df, None
    with open(os.path.join(RES, f"{exp}_hist.json")) as f:
        return df, json.load(f)


def num(x, casas=0):
    """Número no formato brasileiro com espaço fino nos milhares."""
    s = f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", "\\,")
    return s


def pval(p):
    if p < 1e-4:
        return "$<10^{-4}$"
    return num(p, 4 if p < 0.01 else 3)


def escrever(nome, linhas):
    with open(os.path.join(TAB, nome), "w") as f:
        f.write("\n".join(linhas) + "\n")


def curva_mediana(historicos, orc, pontos=400):
    """Mediana entre execuções do melhor h (função escada) numa grade de x."""
    xs = np.unique(np.geomspace(1, orc, pontos).astype(int))
    ys = []
    for hist in historicos:
        a = np.array(hist)
        idx = np.searchsorted(a[:, 0], xs, side="right") - 1
        y = np.where(idx >= 0, a[np.maximum(idx, 0), 1], np.nan)
        ys.append(y)
    return xs, np.nanmedian(np.array(ys), axis=0)


resumo = {}

# ----------------------------------------------------------------- T2 / tempo
t2, _ = ler("t2")
tempo = pd.read_csv(os.path.join(RES, "tempo.csv"))
VIZ = ["swap", "insertion", "2opt"]
NOME_VIZ = {"swap": "Swap", "insertion": "Insertion", "2opt": "2-opt"}
TAM_VIZ = {"swap": "$n(n-1)/2$", "insertion": "$(n-1)^2$", "2opt": "$n(n-1)/2$"}
lin = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
       r" & & \multicolumn{3}{c}{mediana do melhor $h$ (10 exec.)} & "
       r"\multicolumn{3}{c}{$\mu$s por sucessor} \\",
       r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
       r"Instância & Vizinhança & Best & First & $p$ (First) & completa & "
       r"incremental & ganho \\", r"\midrule"]
for i in INST:
    sub = t2[t2.inst == i]
    first = {v: sub[(sub.viz == v) & (sub.alg == "gd_first")].melhor_h for v in VIZ}
    melhor = min(VIZ, key=lambda v: first[v].median())
    for k, v in enumerate(VIZ):
        b = sub[(sub.viz == v) & (sub.alg == "gd_best")].melhor_h.median()
        f = first[v].median()
        p = "--" if v == melhor else pval(mannwhitneyu(first[v], first[melhor]).pvalue)
        tm = tempo[(tempo.inst == i) & (tempo.viz == v)].iloc[0]
        fs = rf"\textbf{{{num(f)}}}" if v == melhor else num(f)
        lin.append(f"{NOME_INST[i] if k == 0 else ''} & {NOME_VIZ[v]} & {num(b)} & "
                   f"{fs} & {p} & {num(tm.us_descida_completa, 1)} & "
                   f"{num(tm.us_descida_incremental, 1)} & "
                   f"{num(tm.ganho_descida, 2)}$\\times$ \\\\")
    if i != INST[-1]:
        lin.append(r"\midrule")
lin += [r"\bottomrule", r"\end{tabular}"]
escrever("t2.tex", lin)

lin = [r"\begin{tabular}{lrrr}", r"\toprule",
       r"Vizinhança & tamanho & $n=50$ & $n=150$ \\", r"\midrule"]
for v in VIZ:
    lin.append(f"{NOME_VIZ[v]} & {TAM_VIZ[v]} & {num(49**2 if v == 'insertion' else 1225)}"
               f" & {num(149**2 if v == 'insertion' else 11175)} \\\\")
lin += [r"\bottomrule", r"\end{tabular}"]
escrever("tam_vizinhanca.tex", lin)

# ------------------------------------------------------------- T3: principal
t3, h3 = ler("t3", historico=True)
lin = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
       r"Instância & Algoritmo & média $\pm$ desvio & mínimo & mediana & "
       r"gap médio & tempo (s) & passos \\", r"\midrule"]
stats = {}
for i in INST:
    for k, a in enumerate(ALGS):
        s = t3[(t3.inst == i) & (t3.alg == a)]
        x = s.melhor_h
        stats[i, a] = x
        gap = f"{num(100 * (x.mean() - H_ESTRELA) / H_ESTRELA, 1)}\\%" \
            if i == "pequena" else "--"
        passos = num(s.passos.median()) if a.startswith("gd") else "--"
        lin.append(f"{NOME_INST[i] if k == 0 else ''} & {NOME_ALG[a]} & "
                   f"{num(x.mean())} $\\pm$ {num(x.std())} & {num(x.min())} & "
                   f"{num(x.median())} & {gap} & {num(s.tempo_s.mean(), 1)} & "
                   f"{passos} \\\\")
    if i != INST[-1]:
        lin.append(r"\midrule")
lin += [r"\bottomrule", r"\end{tabular}"]
escrever("principal.tex", lin)
resumo["principal"] = {f"{i}/{a}": {"media": float(x.mean()), "dp": float(x.std()),
                                    "min": int(x.min()), "mediana": float(x.median())}
                       for (i, a), x in stats.items()}

# Testes de Mann-Whitney (bilateral) para as comparações citadas no texto.
PARES = [("sa", "gd_first"), ("sa", "gd_aleatorio"), ("gd_first", "gd_aleatorio"),
         ("gd_first", "gd_best"), ("sa", "ga"), ("gd_first", "ga")]
lin = [r"\begin{tabular}{l" + "r" * len(INST) + "}", r"\toprule",
       "Comparação & " + " & ".join(NOME_INST[i] for i in INST) + r" \\", r"\midrule"]
resumo["p"] = {}
for a, b in PARES:
    cel = []
    for i in INST:
        xa, xb = stats[i, a], stats[i, b]
        if (xa == xa.iloc[0]).all() and (xb == xa.iloc[0]).all():
            cel.append("idênticos")
            continue
        p = mannwhitneyu(xa, xb).pvalue
        resumo["p"][f"{i}/{a}-{b}"] = p
        sinal = "<" if xa.median() < xb.median() else (">" if xa.median() > xb.median() else "=")
        cel.append(f"{pval(p)} ({sinal})")
    lin.append(f"{NOME_ALG[a]} vs {NOME_ALG[b]} & " + " & ".join(cel) + r" \\")
lin += [r"\bottomrule", r"\end{tabular}"]
escrever("testes.tex", lin)

# Ótimos locais na instância pequena
resumo["otimos_locais"] = {}
for a in ["gd_best", "gd_first", "gd_aleatorio"]:
    ids = t3[(t3.inst == "pequena") & (t3.alg == a)].index
    ots = [(hh, tuple(s)) for k in ids for hh, s in h3[f"t3-{k}"]["otimos"]]
    por_exec = [sum(hh == H_ESTRELA for hh, _ in h3[f"t3-{k}"]["otimos"]) > 0
                for k in ids]
    distintos = {s for _, s in ots}
    resumo["otimos_locais"][a] = {
        "restarts": len(ots), "distintos": len(distintos),
        "valores_distintos": len({hh for hh, _ in ots}),
        "frac_restarts_otimo": float(np.mean([hh == H_ESTRELA for hh, _ in ots])),
        "execucoes_com_otimo": int(sum(por_exec)),
        "distintos_com_h_estrela": len({s for hh, s in ots if hh == H_ESTRELA})}
for a in ["sa", "ga"]:
    resumo["otimos_locais"][a] = {
        "execucoes_com_otimo": int((stats["pequena", a] == H_ESTRELA).sum())}

# Histograma dos valores dos ótimos locais (GD first) na pequena
ids = t3[(t3.inst == "pequena") & (t3.alg == "gd_first")].index
vals = np.array([hh for k in ids for hh, _ in h3[f"t3-{k}"]["otimos"]])
uv, cnt = np.unique(vals, return_counts=True)
fig, ax = plt.subplots(figsize=(6.2, 2.0))
ax.bar(range(len(uv)), cnt / cnt.sum() * 100, color=COR["gd_first"], width=0.8)
ax.set_xticks(range(len(uv)), [str(v) for v in uv], rotation=45, fontsize=7)
ax.set_xlabel("valor de $h$ do ótimo local")
ax.set_ylabel("% dos restarts")
ax.grid(axis="x", visible=False)
fig.savefig(os.path.join(FIG, "otimos_locais.pdf"))
plt.close(fig)

# ---------------------------------------------------------------- boxplots
BOX = ["gd_first", "gd_aleatorio", "sa", "ga"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
for ax, i in zip(axs, INST):
    dados = [stats[i, a] for a in BOX]
    bp = ax.boxplot(dados, widths=0.55, patch_artist=True,
                    medianprops={"color": TINTA, "linewidth": 1.4},
                    flierprops={"marker": "o", "markersize": 3,
                                "markeredgecolor": TINTA2})
    for patch, a in zip(bp["boxes"], BOX):
        patch.set(facecolor=COR[a] + "55", edgecolor=COR[a], linewidth=1.2)
    ax.axhline(REFERENCIA[i], color=TINTA2, linestyle=":", linewidth=1)
    ax.set_xticks(range(1, len(BOX) + 1), [NOME_ALG[a] for a in BOX],
                  rotation=30, fontsize=7.5)
    ax.set_title(TITULO_INST[i], fontsize=9)
    ax.grid(axis="x", visible=False)
axs[0].set_ylabel("melhor $h$ encontrado")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "boxplots.pdf"))
plt.close(fig)

# ----------------------------------------------------- curvas de convergência
CONV = ["gd_first", "gd_aleatorio", "sa", "ga"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
for ax, i in zip(axs, INST):
    for a in CONV:
        ids = t3[(t3.inst == i) & (t3.alg == a)].index
        xs, ys = curva_mediana([h3[f"t3-{k}"]["historico"] for k in ids], ORCAMENTO[i])
        ax.plot(xs, ys, color=COR[a], label=NOME_ALG[a])
        if a == "ga":  # média da população (mediana entre execuções)
            gs = [np.array(h3[f"t3-{k}"]["geracoes"]) for k in ids]
            ng = min(len(g) for g in gs)
            med = np.median([g[:ng, 2] for g in gs], axis=0)
            ax.plot(gs[0][:ng, 0], med, color=COR[a], linestyle="--",
                    linewidth=1.2, label="GA (média da pop.)")
    ax.axhline(REFERENCIA[i], color=TINTA2, linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(TITULO_INST[i], fontsize=9)
    ax.set_xlabel("avaliações de $h$")
axs[0].set_ylabel("melhor $h$ (mediana)")
fig.tight_layout()
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=5,
           fontsize=7.5, bbox_to_anchor=(0.5, -0.08))
fig.savefig(os.path.join(FIG, "convergencia.pdf"))
plt.close(fig)

# ------------------------------------------------- SA: taxa de aceitação
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
resumo["sa"] = {}
for ax, i in zip(axs, INST):
    ids = t3[(t3.inst == i) & (t3.alg == "sa")].index
    series = [np.array(h3[f"t3-{k}"]["serie_aceitacao"]) for k in ids]
    m = min(len(s) for s in series)
    x = series[0][:m, 0] / ORCAMENTO[i]
    taxa = np.array([s[:m, 2] for s in series])
    ax.fill_between(x, np.percentile(taxa, 10, axis=0) * 100,
                    np.percentile(taxa, 90, axis=0) * 100,
                    color=COR["sa"], alpha=0.2, linewidth=0)
    ax.plot(x, np.median(taxa, axis=0) * 100, color=COR["sa"])
    ax.set_title(TITULO_INST[i], fontsize=9)
    ax.set_xlabel("fração do orçamento")
    t0 = t3[(t3.inst == i) & (t3.alg == "sa")].t0
    resumo["sa"][i] = {"t0_mediano": float(t0.median()),
                       "aceit_inicial": float(np.median(taxa[:, :5])),
                       "aceit_final": float(np.median(taxa[:, -10:]))}
axs[0].set_ylabel("aceitação de pioras (%)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "sa_aceitacao.pdf"))
plt.close(fig)

# ------------------------------------------------- GA: diversidade
def suavizar(y, k=25):
    return pd.Series(y).rolling(k, center=True, min_periods=1).mean().to_numpy()


fig, axs = plt.subplots(2, 3, figsize=(7.2, 3.6), sharex=True)
resumo["ga"] = {}
for c, i in enumerate(INST):
    ids = t3[(t3.inst == i) & (t3.alg == "ga")].index
    gs = [np.array(h3[f"t3-{k}"]["geracoes"]) for k in ids]
    ng = min(len(g) for g in gs)
    x = gs[0][:ng, 0] / ORCAMENTO[i]
    distintos = np.median([g[:ng, 3] for g in gs], axis=0)
    dist = np.median([g[:ng, 4] for g in gs], axis=0)
    axs[0, c].plot(x, suavizar(distintos), color=COR["ga"])
    axs[1, c].plot(x, suavizar(dist), color=COR["ga"])
    axs[0, c].set_ylim(0, 52)
    axs[1, c].set_ylim(0, 1.02)
    axs[0, c].set_title(TITULO_INST[i], fontsize=9)
    axs[1, c].set_xlabel("fração do orçamento")
    resumo["ga"][i] = {"distintos_final": float(distintos[-1]),
                       "hamming_final": float(dist[-1]), "geracoes": int(ng),
                       "gap_media_melhor_final": float(np.median(
                           [(g[ng - 1, 2] - g[ng - 1, 1]) / g[ng - 1, 1] for g in gs]))}
axs[0, 0].set_ylabel("distintos (de 50)")
axs[1, 0].set_ylabel("Hamming médio")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "ga_diversidade.pdf"))
plt.close(fig)

# ----------------------------------------------------- experimento extra
ex, _ = ler("extra")
ROT = ["ox", "pmx", "sem_cruzamento"]
NOME_ROT = {"ox": "OX", "pmx": "PMX", "sem_cruzamento": "sem cruzamento ($P_c=0$)"}
lin = [r"\begin{tabular}{llrrr}", r"\toprule",
       r"Instância & GA & média $\pm$ desvio & mediana & $p$ vs sem cruzamento \\",
       r"\midrule"]
resumo["extra"] = {}
for i in INST:
    base = ex[(ex.inst == i) & (ex.rotulo == "sem_cruzamento")].melhor_h
    for k, r in enumerate(ROT):
        x = ex[(ex.inst == i) & (ex.rotulo == r)].melhor_h
        p = "--" if r == "sem_cruzamento" else pval(mannwhitneyu(x, base).pvalue)
        if r != "sem_cruzamento":
            resumo["extra"][f"{i}/{r}"] = mannwhitneyu(x, base).pvalue
        lin.append(f"{NOME_INST[i] if k == 0 else ''} & {NOME_ROT[r]} & "
                   f"{num(x.mean())} $\\pm$ {num(x.std())} & {num(x.median())} & {p} \\\\")
    if i != INST[-1]:
        lin.append(r"\midrule")
lin += [r"\bottomrule", r"\end{tabular}"]
escrever("extra.tex", lin)

# ------------------------------------------------------- calibrações (anexo)
for exp, nome in (("calib_sa", "calib_sa.tex"), ("calib_ga", "calib_ga.tex")):
    c, _ = ler(exp)
    g = c.groupby(["inst", "rotulo"]).melhor_h.median().reset_index()
    lin = [r"\begin{tabular}{l" + "r" * len(INST) + "}", r"\toprule",
           "Configuração & " + " & ".join(NOME_INST[i] for i in INST) + r" \\",
           r"\midrule"]
    for rot in sorted(g.rotulo.unique()):
        cel = []
        for i in INST:
            v = g[(g.inst == i) & (g.rotulo == rot)].melhor_h.iloc[0]
            best = g[g.inst == i].melhor_h.min()
            cel.append(rf"\textbf{{{num(v)}}}" if v == best else num(v))
        rot_tex = rot.replace("chi0", r"$\chi_0$").replace("tf", "$T_f$") \
            .replace("pm", "$P_m$").replace("mut", "mut.")
        lin.append(rot_tex + " & " + " & ".join(cel) + r" \\")
    lin += [r"\bottomrule", r"\end{tabular}"]
    escrever(nome, lin)

with open(os.path.join(RES, "resumo.json"), "w") as f:
    json.dump(resumo, f, indent=1, ensure_ascii=False)
print(json.dumps(resumo, indent=1, ensure_ascii=False))
