"""Escolha do corte de decisão (etapa 5), feita na VALIDAÇÃO.

Treina a configuração escolhida na busca (Dropout 0,1 + weight decay 1e-2, seed 42)
e compara três cortes nas probabilidades de validação:

- 0,5, o padrão;
- a prevalência de "Yes" no ajuste, que espelha o desbalanceamento: com 17,5% de
  positivos, um cliente com probabilidade acima da taxa-base já é "mais provável
  que a média" de comprar;
- o corte que maximiza o F1 na validação, como referência.

Corte escolhido: a prevalência, porque deixar escapar um comprador foi tratado
como o erro mais caro (recall alto). O teste NÃO é usado; esse corte é o que
será aplicado nele.

    python docs/projects/classification/code/threshold.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data_prep import load_tensors  # noqa: E402
from train import fit, validation_metrics  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = (BASE_DIR / "../figures").resolve()
OUT_DIR = (BASE_DIR / "../output").resolve()
CHOSEN = {"dropout": 0.1, "weight_decay": 1e-2}


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    data = load_tensors()
    model, _ = fit(data, seed=42, verbose=False, **CHOSEN)

    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(data["X_valid"])).numpy()
    y_true = data["y_valid"].numpy()

    prevalence = data["y_fit"].mean().item()
    precision, recall, cuts = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]  # o último ponto não tem corte
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best_f1_cut = float(cuts[np.argmax(f1)])

    thresholds = {
        "0,5 (padrão)": 0.5,
        f"Prevalência ({prevalence:.4f})": prevalence,
        f"Máx. F1 na validação ({best_f1_cut:.4f})": best_f1_cut,
    }
    rows = []
    for name, t in thresholds.items():
        m = validation_metrics(model, data["X_valid"], data["y_valid"], threshold=t)
        rows.append(
            {
                "Corte": name,
                **{k: round(v * 100, 2) for k, v in m.items() if k not in ("ROC AUC", "PR-AUC")},
                "Previstos 'Yes' (%)": round(float((proba >= t).mean()) * 100, 2),
            }
        )
    table = pd.DataFrame(rows).set_index("Corte")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cuts, precision, label="Precisão")
    ax.plot(cuts, recall, label="Recall")
    ax.plot(cuts, f1, label="F1")
    styles = ["--", ":", "-."]
    for (name, t), style in zip(thresholds.items(), styles):
        ax.axvline(t, color="gray", linestyle=style, label=f"Corte {name}")
    ax.set_xlabel("Corte de decisão (probabilidade de 'Yes')")
    ax.set_ylabel("Valor na validação")
    ax.set_xlim(0, 1)
    ax.set_title("Figura 2 - Precisão, recall e F1 por corte (validação)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02-corte-validacao.svg")

    lines = [
        "=== Escolha do corte de decisão (validação; teste não usado) ===",
        "Modelo: Dropout 0,1 + weight decay 1e-2, 64-32, ReLU, AdamW lr=1e-3, seed 42",
        f"Prevalência de 'Yes' no ajuste: {prevalence * 100:.2f}% "
        f"(na validação: {y_true.mean() * 100:.2f}%)",
        "ROC AUC e PR-AUC não dependem do corte: "
        f"{validation_metrics(model, data['X_valid'], data['y_valid'])['PR-AUC'] * 100:.2f} (PR-AUC)",
        "",
        "Métricas na validação por corte (em %):",
        table.to_string(),
        "",
        f"Corte escolhido: prevalência de 'Yes' no ajuste ({prevalence:.4f}).",
    ]
    text = "\n".join(lines)
    print(text)
    (OUT_DIR / "corte_output.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
