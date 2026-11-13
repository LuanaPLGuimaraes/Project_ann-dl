"""Baselines e redes da 2ª rodada com o corte escolhido (prevalência), na VALIDAÇÃO.

Recalcula as métricas que dependem do corte (acurácia, precisão, recall, F1...)
usando o corte = prevalência de "Yes" no ajuste, escolhido em threshold.py, e
repete com 0,5 para comparação. PR-AUC e ROC AUC não dependem do corte.
Salva também a Figura 3 (curvas ROC), usada para comparar com outros grupos
da mesma competição.

- Baselines: os mesmos modelos de baseline.py, no mesmo split de validação.
- Redes: as 15 configurações de search2.py e a escolhida na 1ª rodada, seed 42.

O teste NÃO é usado.

    python docs/projects/classification/code/compare_threshold.py
"""
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from baseline import VALIDATION_SIZE, build_models  # noqa: E402
from data_prep import load_tensors
from eda import RANDOM_STATE, load_split
from preprocess import prepare_xy  # type: ignore[reportMissingImports]
from search2 import MAX_EPOCHS, build_configs
from train import fit

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = (BASE_DIR / "../output").resolve()
FIG_DIR = (BASE_DIR / "../figures").resolve()
CHOSEN_ROUND1 = ("Dropout 0,1 + WD 1e-2 (escolhida, 1ª rodada)", {"dropout": 0.1, "weight_decay": 1e-2})


def metrics_at(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "Acurácia": accuracy_score(y_true, pred),
        "Acurácia balanceada": balanced_accuracy_score(y_true, pred),
        "Precisão": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, proba),
        "PR-AUC": average_precision_score(y_true, proba),
    }


def baseline_probas() -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Treina as baselines como em baseline.py e devolve as probabilidades de validação."""
    train_df, _test_df = load_split()
    X, y = prepare_xy(train_df, use_subsidy=True)
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X, y, test_size=VALIDATION_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    probas = {}
    for name, model in build_models(use_subsidy=True).items():
        print(f"Baseline: {name} ...", flush=True)
        model.fit(X_fit, y_fit)
        probas[name] = model.predict_proba(X_valid)[:, 1]
    return probas, y_valid.to_numpy()


def network_probas(data: dict) -> dict[str, np.ndarray]:
    configs = [CHOSEN_ROUND1] + [(f"Rede: {n}", kw) for n, kw in build_configs()]
    probas = {}
    for i, (name, kwargs) in enumerate(configs, 1):
        started = time.time()
        print(f"[{i}/{len(configs)}] {name} ...", flush=True)
        model, _ = fit(data, seed=42, verbose=False, max_epochs=MAX_EPOCHS, **kwargs)
        model.eval()
        with torch.no_grad():
            probas[name] = torch.sigmoid(model(data["X_valid"])).numpy()
        print(f"    {time.time() - started:.0f}s", flush=True)
    return probas


def plot_roc(probas: dict, y_true: np.ndarray, prevalence: float) -> None:
    """Figura 3: curvas ROC das baselines e das redes de destaque, com zoom.

    Marca na rede escolhida os pontos de operação dos cortes 0,5 e prevalência.
    """
    shown = {
        "Regressão logística": "Regressão logística",
        "Random forest": "Random forest",
        CHOSEN_ROUND1[0]: "Rede escolhida (Dropout 0,1 + WD 1e-2)",
        "Rede: Tanh | 128-64-32 | lr 3e-04": "Rede: tanh, 128-64-32, lr 3e-4",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, zoom in zip(axes, (False, True)):
        for key, label in shown.items():
            fpr, tpr, _ = roc_curve(y_true, probas[key])
            auc = roc_auc_score(y_true, probas[key]) * 100
            ax.plot(fpr, tpr, linewidth=1.4, label=f"{label} (AUC {auc:.2f})")
        chosen = probas[CHOSEN_ROUND1[0]]
        for t, marker, name in ((0.5, "o", "0,5"), (prevalence, "s", f"{prevalence:.3f}")):
            pred = chosen >= t
            tpr_t = pred[y_true == 1].mean()
            fpr_t = pred[y_true == 0].mean()
            ax.plot(fpr_t, tpr_t, marker, color="black", markersize=7,
                    label=f"Rede escolhida, corte {name} (FPR {fpr_t:.2f}, recall {tpr_t:.2f})")
        ax.plot([0, 1], [0, 1], linestyle=":", color="gray", label="Acaso (AUC 50)")
        ax.set_xlabel("Taxa de falsos positivos (FPR)")
        ax.set_ylabel("Taxa de verdadeiros positivos (recall)")
        if zoom:
            ax.set_xlim(0, 0.3)
            ax.set_ylim(0.6, 1.0)
            ax.set_title("Zoom: FPR de 0 a 0,3")
        else:
            ax.set_title("Curva ROC completa")
            ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Figura 3 - Curvas ROC na validação")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03-roc-validacao.svg")
    plt.close(fig)


def table_at(probas: dict, y_true: np.ndarray, threshold: float) -> pd.DataFrame:
    rows = {name: metrics_at(y_true, p, threshold) for name, p in probas.items()}
    table = (pd.DataFrame.from_dict(rows, orient="index") * 100).round(2)
    table.index.name = "Modelo"
    return table.sort_values("F1", ascending=False)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    data = load_tensors()
    prevalence = data["y_fit"].mean().item()

    base_probas, y_valid_base = baseline_probas()
    y_valid = data["y_valid"].numpy()
    # Mesmo split e mesma ordem de linhas: as baselines e as redes veem a mesma validação.
    assert np.array_equal(y_valid_base.astype(np.float32), y_valid)

    probas = {**base_probas, **network_probas(data)}

    plot_roc(probas, y_valid, prevalence)

    cols = ["Acurácia", "Acurácia balanceada", "Precisão", "Recall", "F1", "ROC AUC", "PR-AUC"]
    at_prev = table_at(probas, y_valid, prevalence)
    at_half = table_at(probas, y_valid, 0.5)
    delta = (at_prev[["Precisão", "Recall", "F1"]] - at_half[["Precisão", "Recall", "F1"]]).round(2)
    delta = delta.loc[at_prev.index]
    delta.columns = [f"Δ {c}" for c in delta.columns]

    lines = [
        "=== Baselines e redes com o corte escolhido (validação; teste não usado) ===",
        f"Corte = prevalência de 'Yes' no ajuste: {prevalence:.4f} (antes: 0,5)",
        "Redes: seed 42, max_epochs = 200, patience = 10. Ordenado pelo F1 no novo corte.",
        "O Dummy prevê uma probabilidade constante (a taxa de 'Yes'), que fica logo abaixo do corte: marca todos como 'No'.",
        "",
        f"Métricas com corte {prevalence:.4f} (em %):",
        at_prev[cols].to_string(),
        "",
        "Variação em relação ao corte 0,5 (pontos):",
        delta.to_string(),
        "",
        "Métricas com corte 0,5 (em %), para referência:",
        at_half[cols].to_string(),
    ]
    text = "\n".join(lines)
    print(text)
    (OUT_DIR / "comparacao_corte_output.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
