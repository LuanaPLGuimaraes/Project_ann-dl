"""Treino da MLP com early stopping (PyTorch).

Monitora a loss de validação, guarda os pesos da melhor época e os restaura no
fim. Registra loss de treino, loss de validação e PR-AUC de validação por época.

Treina a rede base e salva a Figura 1 e os números::

    python docs/projects/classification/code/train.py
"""
import copy
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn

matplotlib.use("Agg")  # só salva figuras, sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402

from data_prep import RANDOM_STATE, load_tensors  # noqa: E402
from network import MLP, count_parameters  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = (BASE_DIR / "../figures").resolve()
OUT_DIR = (BASE_DIR / "../output").resolve()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 8192,
    pos_weight: float | None = None,
):
    """Loss (BCE, com o mesmo peso de classe do treino, se houver) e PR-AUC, sem ajustar nada."""
    loss_fn = nn.BCEWithLogitsLoss(
        reduction="sum",
        pos_weight=None if pos_weight is None else torch.tensor([pos_weight]),
    )
    model.eval()
    total_loss = 0.0
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            logits = model(X[start:start + batch_size])
            total_loss += loss_fn(logits, y[start:start + batch_size]).item()
            probabilities.append(torch.sigmoid(logits))
    scores = torch.cat(probabilities).numpy()
    return total_loss / len(X), average_precision_score(y.numpy(), scores)


def validation_metrics(model: nn.Module, X: torch.Tensor, y: torch.Tensor, threshold: float = 0.5) -> dict:
    """Métricas de classificação em um conjunto, com um corte de decisão."""
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X)).numpy()
    y_true = y.numpy()
    predictions = (probabilities >= threshold).astype(int)
    return {
        "Acurácia": accuracy_score(y_true, predictions),
        "Acurácia balanceada": balanced_accuracy_score(y_true, predictions),
        "Precisão": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
    }


def fit(
    data: dict,
    *,
    hidden_dims: tuple[int, ...] = (64, 32),
    activation: str = "relu",
    dropout: float = 0.0,
    batch_norm: bool = False,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 1024,
    max_epochs: int = 100,
    patience: int = 10,
    pos_weight: float | None = None,
    seed: int = RANDOM_STATE,
    verbose: bool = True,
):
    """Treina uma MLP e devolve (modelo com os melhores pesos, histórico)."""
    set_seed(seed)  # antes de criar o modelo, para a inicialização ser reprodutível
    model = MLP(data["n_inputs"], hidden_dims, activation, dropout, batch_norm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=None if pos_weight is None else torch.tensor([pos_weight])
    )

    X, y = data["X_fit"], data["y_fit"]
    shuffler = torch.Generator().manual_seed(seed)
    history = {"train_loss": [], "val_loss": [], "val_pr_auc": []}
    best_val_loss, best_epoch, best_state, waited = float("inf"), 0, None, 0

    for epoch in range(1, max_epochs + 1):
        started = time.time()
        model.train()
        order = torch.randperm(len(X), generator=shuffler)
        running = 0.0
        for start in range(0, len(X), batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = train_loss_fn(model(X[idx]), y[idx])
            loss.backward()
            optimizer.step()
            running += loss.item() * len(idx)
        train_loss = running / len(X)

        val_loss, val_pr_auc = evaluate(
            model, data["X_valid"], data["y_valid"], pos_weight=pos_weight
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_pr_auc)

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, waited = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1

        if verbose:
            print(
                f"época {epoch:3d} | treino {train_loss:.4f} | val {val_loss:.4f} "
                f"| PR-AUC val {val_pr_auc:.4f} | {time.time() - started:.1f}s"
            )
        if waited >= patience:
            break

    model.load_state_dict(best_state)  # volta à melhor época, não à última
    history["best_epoch"] = best_epoch
    return model, history


def run_base() -> None:
    """Treina a rede base até o early stopping e salva a Figura 1 e os números."""
    FIG_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    data = load_tensors()
    model, h = fit(data)  # base: 64-32, ReLU, AdamW lr=1e-3, batch=1024, sem regularização
    epochs = list(range(1, len(h["val_loss"]) + 1))
    best = h["best_epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, h["train_loss"], label="Treino")
    ax1.plot(epochs, h["val_loss"], label="Validação")
    ax1.axvline(best, linestyle="--", color="gray", label=f"Melhor época ({best})")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss (BCE)")
    ax1.set_title("Loss por época")
    ax1.legend()

    ax2.plot(epochs, h["val_pr_auc"], label="PR-AUC (validação)")
    ax2.axvline(best, linestyle="--", color="gray", label=f"Melhor época ({best})")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("PR-AUC")
    ax2.set_title("PR-AUC de validação por época")
    ax2.legend()

    fig.suptitle("Figura 1 - Rede base (64-32, ReLU, AdamW): curvas de treino")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01-curvas-treino-base.svg")

    best_pr = max(h["val_pr_auc"])
    lines = [
        "=== Rede base (64-32, ReLU, AdamW lr=1e-3, batch=1024, sem regularização) ===",
        f"Parâmetros: {count_parameters(model):,}",
        f"Épocas treinadas: {len(epochs)} | melhor época (loss de validação): {best}",
        f"Loss de validação na melhor época: {h['val_loss'][best - 1]:.4f}",
        f"Loss de treino na melhor época: {h['train_loss'][best - 1]:.4f}",
        f"PR-AUC de validação na melhor época: {h['val_pr_auc'][best - 1]:.4f}",
        f"PR-AUC de validação máxima: {best_pr:.4f} (época {h['val_pr_auc'].index(best_pr) + 1})",
    ]
    metrics = validation_metrics(model, data["X_valid"], data["y_valid"])
    lines.append("Métricas na validação (melhores pesos, corte 0,5), em %:")
    lines += [f"  {name}: {value * 100:.2f}" for name, value in metrics.items()]

    text = "\n".join(lines)
    print(text)
    (OUT_DIR / "treino_base_output.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_base()