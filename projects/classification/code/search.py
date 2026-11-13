"""Busca de regularização e hiperparâmetros (etapa 4), escolhida pela VALIDAÇÃO.

Treina 12 configurações com seed 42, ordena pela PR-AUC de validação e confirma
as 3 melhores com 3 seeds. O teste NÃO é usado.

    python docs/projects/classification/code/search.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_prep import load_tensors
from network import count_parameters
from train import fit

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = (BASE_DIR / "../output").resolve()
SEEDS = (42, 43, 44)


def build_configs(pos_weight: float) -> list[tuple[str, dict]]:
    return [
        ("Base (64-32, sem regularização)", {}),
        ("Weight decay 1e-4", {"weight_decay": 1e-4}),
        ("Weight decay 1e-2", {"weight_decay": 1e-2}),
        ("Dropout 0,1", {"dropout": 0.1}),
        ("Dropout 0,3", {"dropout": 0.3}),
        ("BatchNorm", {"batch_norm": True}),
        ("Dropout 0,1 + weight decay 1e-2", {"dropout": 0.1, "weight_decay": 1e-2}),
        (f"Peso da classe rara ({pos_weight:.2f})", {"pos_weight": pos_weight}),
        ("Mais larga (128-64)", {"hidden_dims": (128, 64)}),
        ("Menor (32-16)", {"hidden_dims": (32, 16)}),
        ("Ativação tanh", {"activation": "tanh"}),
        ("Learning rate 3e-4", {"lr": 3e-4}),
    ]


def run_one(data: dict, name: str, kwargs: dict, seed: int = 42) -> dict:
    started = time.time()
    model, h = fit(data, seed=seed, verbose=False, **kwargs)
    best = h["best_epoch"]
    return {
        "Configuração": name,
        "Parâmetros": count_parameters(model),
        "Melhor época": best,
        "Loss treino": h["train_loss"][best - 1],
        "Loss val": h["val_loss"][best - 1],
        "PR-AUC val": h["val_pr_auc"][best - 1],
        "Tempo (s)": time.time() - started,
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    data = load_tensors()
    rate = data["y_fit"].mean().item()
    pos_weight = (1 - rate) / rate
    configs = build_configs(pos_weight)

    rows = []
    for i, (name, kwargs) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {name} ...", flush=True)
        row = run_one(data, name, kwargs)
        print(
            f"    PR-AUC val {row['PR-AUC val']:.4f} | melhor época {row['Melhor época']} "
            f"| {row['Tempo (s)']:.0f}s",
            flush=True,
        )
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("PR-AUC val", ascending=False).reset_index(drop=True)
    table.index += 1

    # Confirmação: as 3 melhores, com 3 seeds (a seed 42 já foi rodada na busca).
    kwargs_by_name = dict(configs)
    confirm_rows = []
    for name in table["Configuração"].head(3):
        print(f"Confirmando com 3 seeds: {name} ...", flush=True)
        first = float(table.loc[table["Configuração"] == name, "PR-AUC val"].iloc[0])
        scores = [first] + [
            run_one(data, name, kwargs_by_name[name], seed=s)["PR-AUC val"] for s in SEEDS[1:]
        ]
        confirm_rows.append(
            {
                "Configuração": name,
                "PR-AUC val média (%)": round(float(np.mean(scores)) * 100, 2),
                "Desvio (pontos)": round(float(np.std(scores, ddof=1)) * 100, 2),
                "Por seed (%)": " / ".join(f"{s * 100:.2f}" for s in scores),
            }
        )
    confirm = pd.DataFrame(confirm_rows).sort_values("PR-AUC val média (%)", ascending=False)
    confirm = confirm.reset_index(drop=True)
    confirm.index += 1

    show = table.copy()
    show["PR-AUC val"] = (show["PR-AUC val"] * 100).round(2)
    show["Loss treino"] = show["Loss treino"].round(4)
    show["Loss val"] = show["Loss val"].round(4)
    show["Tempo (s)"] = show["Tempo (s)"].round(0).astype(int)

    lines = [
        "=== Busca de configurações (seed 42, escolha pela validação) ===",
        f"Peso da classe rara usado: {pos_weight:.2f} (proporção No/Yes no ajuste)",
        "Referência (baselines, validação): regressão logística 73,72 | random forest 73,80",
        "Nota: a loss com peso de classe não é comparável com a das demais configurações.",
        "",
        "PR-AUC de validação na melhor época (em %), ordenada:",
        show.to_string(),
        "",
        "=== Confirmação das 3 melhores com 3 seeds (42, 43, 44) ===",
        confirm.to_string(),
        "",
        f"Configuração escolhida (maior média nas 3 seeds): {confirm.iloc[0]['Configuração']}",
    ]
    text = "\n".join(lines)
    print(text)
    (OUT_DIR / "busca_output.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()