"""Análise bivariada/multivariada (etapa 3), com no máximo 3 figuras por item.

3A num x num      -> 2 figuras (matriz de correlação + scatter dos pares mais correlacionados)
3B cat x alvo     -> 3 figuras (2 categóricas por figura; as duas suspeitas de vazamento juntas)
3C num x cat      -> 3 figuras (boxplots agrupados)

Saída de texto com todos os números: ../output/bivariate_output.txt
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from eda import (CATEGORICAL_COLS, NUMERIC_COLS, OUTPUT_PATH, RANDOM_STATE,
                 TARGET, Report, load_split, save_fig)

CORR_METHOD = "spearman"   # justificar no relatório (veja a comparação com Pearson no .txt)
N_SCATTER_PAIRS = 3
SCATTER_SAMPLE = 5_000

# 3B: 6 categóricas em 3 figuras. As duas suspeitas de vazamento ficam juntas.
CAT_TARGET_FIGS = [
    ["Subsidy_Available", "Home_Charging_Possible"],
    ["Range_Anxiety_Level", "City_Type"],
    ["Current_Car_Type", "Gender"],
]

# 3C: (numérica, categórica). Escolha depois de olhar o .txt, os 3 mais informativos.
BOXPLOTS = [
    ("Annual_Income_USD", "City_Type"),
    ("Daily_Commute_km", "Current_Car_Type"),
    ("Annual_Income_USD", TARGET),   # Range_Anxiety_Level foi descartada; renda x alvo é mais informativa
]

FIG_START = 5  # 3A: 5-6 | 3B: 7-9 | 3C: 10-12
BIVARIATE_OUTPUT = OUTPUT_PATH.parent / "bivariate_output.txt"


def to_binary(df):
    df = df.copy()
    if df[TARGET].dtype == object or str(df[TARGET].dtype).startswith("str"):
        df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})
    df[TARGET] = df[TARGET].astype(int)
    return df


# ---------- 3A ----------
def correlation_section(df, report, fig_num):
    report.section("3A. Correlação numérica x numérica")
    pearson = df[NUMERIC_COLS].corr(method="pearson")
    spearman = df[NUMERIC_COLS].corr(method="spearman")
    corr = spearman if CORR_METHOD == "spearman" else pearson

    report.add("Pearson:\n" + pearson.round(3).to_string())
    report.add("\nSpearman:\n" + spearman.round(3).to_string())

    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    ranked = upper.reindex(upper.abs().sort_values(ascending=False).index)
    report.add(f"\nPares mais correlacionados ({CORR_METHOD}):")
    for (a, b), v in ranked.head(5).items():
        report.add(f"  {a} x {b}: {v:+.3f}")
    report.add(f"Pares com |r| >= 0.7 (redundantes): {int((ranked.abs() >= 0.7).sum())}")

    tgt = df[NUMERIC_COLS + [TARGET]].corr(method=CORR_METHOD)[TARGET].drop(TARGET)
    report.add(f"\nCorrelação de cada numérica com o alvo ({CORR_METHOD}):\n" + tgt.round(3).to_string())

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Correlação de {CORR_METHOD.capitalize()}")
    ax.set_title(f"Correlação de {CORR_METHOD.capitalize()} entre numéricas (treino)")
    paths = [save_fig(fig, f"fig{fig_num:02d}-correlacao.svg")]

    pairs = list(ranked.head(N_SCATTER_PAIRS).index)
    sample = df.sample(min(SCATTER_SAMPLE, len(df)), random_state=RANDOM_STATE)
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.2))
    for ax, (a, b) in zip(np.atleast_1d(axes), pairs):
        ax.scatter(sample[a], sample[b], s=4, alpha=0.3, color="#2a78d6",
                   label=f"{corr.loc[a, b]:+.2f} ({CORR_METHOD})")
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title(f"{a} x {b}")
        ax.legend(title="Correlação")
    paths.append(save_fig(fig, f"fig{fig_num + 1:02d}-scatter-pares.svg"))
    return paths


# ---------- 3B ----------
def categorical_vs_target_section(df, report, fig_num):
    report.section("3B. Categóricas x alvo (proporção de Yes por categoria)")
    overall = df[TARGET].mean()
    report.add(f"Taxa geral de Yes no treino: {overall:.2%}")
    paths = []
    for k, group in enumerate(CAT_TARGET_FIGS):
        fig, axes = plt.subplots(1, len(group), figsize=(6 * len(group), 4))
        for ax, col in zip(np.atleast_1d(axes), group):
            stats = df.groupby(col)[TARGET].agg(["mean", "count"]).sort_values("mean")
            report.add(f"\n{col}:\n" + stats.rename(columns={"mean": "taxa_yes", "count": "n"})
                       .round(4).to_string())
            bars = ax.barh(stats.index.astype(str), stats["mean"], color="#2a78d6",
                           label="Taxa de Yes na categoria")
            ax.bar_label(bars, labels=[f"{v:.1%}" for v in stats["mean"]], padding=3)
            ax.axvline(overall, color="gray", linestyle="--", label=f"Média geral ({overall:.1%})")
            ax.set_xlim(0, min(1, stats["mean"].max() * 1.25 + 0.02))
            ax.set_xlabel("Proporção de Yes")
            ax.set_title(f"{TARGET} por {col}")
            ax.legend(loc="lower right")
        paths.append(save_fig(fig, f"fig{fig_num + k:02d}-cat-vs-alvo-{k + 1}.svg"))
    return paths


# ---------- 3C ----------
def boxplot_section(df, report, fig_num):
    report.section("3C. Numérica x categórica (boxplots agrupados)")
    paths = []
    for k, (num, cat) in enumerate(BOXPLOTS):
        order = [c for c in ("Low", "Medium", "High") if c in set(df[cat])] or sorted(df[cat].unique())
        data = [df.loc[df[cat] == c, num] for c in order]
        stats = pd.DataFrame({
            "n": [len(d) for d in data],
            "mediana": [d.median() for d in data],
            "IQR": [d.quantile(0.75) - d.quantile(0.25) for d in data],
            "desvio": [d.std() for d in data],
        }, index=order).round(2)
        report.add(f"\n{num} por {cat}:\n" + stats.to_string())

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        labels = ["No", "Yes"] if cat == TARGET else order
        ax.boxplot(data, tick_labels=labels, flierprops={"markersize": 1, "alpha": 0.2},
                   medianprops={"color": "#eb6834", "linewidth": 2})
        ax.set_xlabel(cat)
        ax.set_ylabel(num)
        ax.set_title(f"{num} por {cat} (treino)")
        ax.legend(handles=[Line2D([0], [0], color="#eb6834", lw=2, label="Mediana")])
        paths.append(save_fig(fig, f"fig{fig_num + k:02d}-box-{num.lower().replace('_', '-')}.svg"))
    return paths


def main():
    train_df, _ = load_split()
    df = to_binary(train_df)
    report = Report(BIVARIATE_OUTPUT)

    figs = []
    figs += correlation_section(df, report, FIG_START)
    figs += categorical_vs_target_section(df, report, FIG_START + 2)
    figs += boxplot_section(df, report, FIG_START + 5)

    report.section("Figuras salvas")
    for path in figs:
        report.add(path)
    report.save()


if __name__ == "__main__":
    main()