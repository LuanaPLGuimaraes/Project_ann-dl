"""Redução de dimensionalidade (etapa 4B): PCA, t-SNE e UMAP.

Usa as features já escalonadas do TREINO (saída de preprocess.py), coloridas pelo alvo.
t-SNE e UMAP rodam em uma amostra estratificada do treino (SAMPLE_SIZE linhas).
Saída de texto: ../output/dimred_output.txt
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from eda import OUTPUT_PATH, RANDOM_STATE, Report, save_fig
from preprocess import run as run_preprocess

SAMPLE_SIZE = 10_000
PERPLEXITIES = (30, 50)
N_NEIGHBORS = (15, 50)
FIG_START = 16  # número da primeira figura desta etapa (ajuste conforme o relatório)

CLASS_STYLE = {0: ("No", "#2a78d6"), 1: ("Yes", "#eb6834")}
DIMRED_OUTPUT = OUTPUT_PATH.parent / "dimred_output.txt"


def scatter_by_target(ax, emb, y, title):
    for cls in (0, 1):  # "Yes" (minoritária) desenhada por cima
        label, color = CLASS_STYLE[cls]
        mask = y == cls
        ax.scatter(emb[mask, 0], emb[mask, 1], s=4, alpha=0.4, c=color, label=label)
    ax.set_title(title)
    ax.legend(title="Will_Buy_EV", markerscale=4)


def run_pca(X_train_t, X_sample, y_sample, names, report, fig_num):
    pca = PCA(random_state=RANDOM_STATE).fit(X_train_t)
    cum = np.cumsum(pca.explained_variance_ratio_)

    report.section("PCA")
    report.add(f"Variância explicada por PC1 + PC2: {cum[1]:.2%}")
    for target in (0.80, 0.90, 0.95):
        report.add(f"Componentes para {target:.0%} da variância: {int(np.searchsorted(cum, target) + 1)}")
    var_df = pd.DataFrame({
        "var_explicada": pca.explained_variance_ratio_,
        "acumulada": cum,
    }, index=[f"PC{i}" for i in range(1, len(cum) + 1)]).round(4)
    report.add("\n" + var_df.to_string())

    loadings = pd.DataFrame(
        pca.components_[:3].T, index=names, columns=["PC1", "PC2", "PC3"]
    ).round(3)
    report.add("\nLoadings (3 primeiros componentes):\n" + loadings.to_string())
    for pc in loadings.columns:
        top = loadings[pc].abs().sort_values(ascending=False).head(3).index
        report.add(f"Maiores |loadings| em {pc}: " + ", ".join(f"{f} ({loadings.loc[f, pc]:+.3f})" for f in top))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    xs = np.arange(1, len(cum) + 1)
    axes[0].bar(xs, pca.explained_variance_ratio_, color="#2a78d6", label="Individual")
    axes[0].plot(xs, cum, color="#eb6834", marker="o", label="Acumulada")
    axes[0].set_xlabel("Componente principal")
    axes[0].set_ylabel("Variância explicada")
    axes[0].set_title("PCA: variância explicada (treino)")
    axes[0].legend()
    emb = pca.transform(X_sample)[:, :2]
    scatter_by_target(axes[1], emb, y_sample, f"PCA: PC1 x PC2 (amostra de {len(y_sample)})")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    return save_fig(fig, f"fig{fig_num:02d}-pca.svg")


def run_tsne(X_sample, y_sample, report, fig_num):
    report.section("t-SNE")
    fig, axes = plt.subplots(1, len(PERPLEXITIES), figsize=(6 * len(PERPLEXITIES), 5))
    for ax, perp in zip(np.atleast_1d(axes), PERPLEXITIES):
        tsne = TSNE(n_components=2, perplexity=perp, init="pca",
                    learning_rate="auto", random_state=RANDOM_STATE)
        emb = tsne.fit_transform(X_sample)
        report.add(f"perplexity={perp}: KL divergence final = {tsne.kl_divergence_:.4f}")
        scatter_by_target(ax, emb, y_sample, f"t-SNE (perplexity={perp})")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    return save_fig(fig, f"fig{fig_num:02d}-tsne.svg")


def run_umap(X_sample, y_sample, report, fig_num):
    report.section("UMAP")
    fig, axes = plt.subplots(1, len(N_NEIGHBORS), figsize=(6 * len(N_NEIGHBORS), 5))
    for ax, nn in zip(np.atleast_1d(axes), N_NEIGHBORS):
        reducer = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=0.1,
                            random_state=RANDOM_STATE)
        emb = reducer.fit_transform(X_sample)
        report.add(f"n_neighbors={nn}, min_dist=0.1: ok")
        scatter_by_target(ax, emb, y_sample, f"UMAP (n_neighbors={nn})")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    return save_fig(fig, f"fig{fig_num:02d}-umap.svg")


def main():
    pre, X_train_t, _, y_train, _ = run_preprocess()
    names = pre.get_feature_names_out()

    n = min(SAMPLE_SIZE, len(y_train))
    _, X_sample, _, y_sample = train_test_split(
        X_train_t, y_train.to_numpy(), test_size=n, stratify=y_train,
        random_state=RANDOM_STATE,
    )

    report = Report(DIMRED_OUTPUT)
    report.section("Amostra para t-SNE/UMAP")
    report.add(f"Amostra estratificada do treino: {len(y_sample)} de {len(y_train)} linhas "
               f"(classe Yes na amostra: {y_sample.mean():.2%}), random_state={RANDOM_STATE}")

    figs = [
        run_pca(X_train_t, X_sample, y_sample, names, report, FIG_START),
        run_tsne(X_sample, y_sample, report, FIG_START + 1),
        run_umap(X_sample, y_sample, report, FIG_START + 2),
    ]
    report.section("Figuras salvas")
    for path in figs:
        report.add(path)
    report.save()


if __name__ == "__main__":
    main()