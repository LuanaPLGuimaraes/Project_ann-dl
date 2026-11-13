"""Segunda rodada da busca (etapa 4), escolhida pela VALIDAÇÃO.

Parte das 3 melhores da primeira rodada (search.py) e testa duas direções:

- learning rate menor (3e-4 e 1e-4): na 1ª rodada a melhor época veio cedo
  (15-30 épocas), então um passo menor pode achar um mínimo melhor;
- mais profundidade: 3 camadas (128-64-32) e 4 camadas (128-64-32-16).

Treina tudo com seed 42, ordena pela PR-AUC de validação e confirma as 3 melhores
com 3 seeds. O teste NÃO é usado. ``max_epochs`` sobe para 200 porque com lr menor
o early stopping demora mais a disparar.

    python docs/projects/classification/code/search2.py
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
MAX_EPOCHS = 200

# As 3 melhores da 1ª rodada (média em 3 seeds, PR-AUC val %): 73,93 / 73,89 / 73,88.
TOP3 = {
    "Dropout 0,1 + WD 1e-2": {"dropout": 0.1, "weight_decay": 1e-2},
    "Tanh": {"activation": "tanh"},
    "Dropout 0,3": {"dropout": 0.3},
}
DEEP3 = (128, 64, 32)
DEEP4 = (128, 64, 32, 16)


def build_configs() -> list[tuple[str, dict]]:
    configs = []
    for name, base in TOP3.items():
        for lr in (3e-4, 1e-4):
            configs.append((f"{name} | lr {lr:.0e}", {**base, "lr": lr}))
    for name, base in TOP3.items():
        configs.append((f"{name} | 128-64-32", {**base, "hidden_dims": DEEP3}))
        configs.append((f"{name} | 128-64-32-16", {**base, "hidden_dims": DEEP4}))
    # Profundidade + lr menor juntos, para ver se os dois efeitos somam.
    for name, base in TOP3.items():
        configs.append(
            (f"{name} | 128-64-32 | lr 3e-04", {**base, "hidden_dims": DEEP3, "lr": 3e-4})
        )
    return configs


def run_one(data: dict, name: str, kwargs: dict, seed: int = 42) -> dict:
    started = time.time()
    model, h = fit(data, seed=seed, verbose=False, max_epochs=MAX_EPOCHS, **kwargs)
    best = h["best_epoch"]
    return {
        "Configuração": name,
        "Parâmetros": count_parameters(model),
        "Épocas": len(h["val_loss"]),
        "Melhor época": best,
        "Loss treino": h["train_loss"][best - 1],
        "Loss val": h["val_loss"][best - 1],
        "PR-AUC val": h["val_pr_auc"][best - 1],
        "Tempo (s)": time.time() - started,
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    data = load_tensors()
    configs = build_configs()

    rows = []
    for i, (name, kwargs) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {name} ...", flush=True)
        row = run_one(data, name, kwargs)
        print(
            f"    PR-AUC val {row['PR-AUC val']:.4f} | melhor época {row['Melhor época']}"
            f"/{row['Épocas']} | {row['Tempo (s)']:.0f}s",
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
        "=== Busca, 2ª rodada (seed 42, escolha pela validação) ===",
        "Variações das 3 melhores da 1ª rodada: lr menor (3e-4, 1e-4) e mais profundidade.",
        f"max_epochs = {MAX_EPOCHS}, patience = 10 (early stopping pela loss de validação).",
        "Referência (1ª rodada, média 3 seeds): Dropout 0,1 + weight decay 1e-2 = 73,93",
        "Referência (baselines, validação): regressão logística 73,72 | random forest 73,80",
        "",
        "PR-AUC de validação na melhor época (em %), ordenada:",
        show.to_string(),
        "",
        "=== Confirmação das 3 melhores com 3 seeds (42, 43, 44) ===",
        confirm.to_string(),
        "",
        f"Melhor da 2ª rodada (maior média nas 3 seeds): {confirm.iloc[0]['Configuração']}",
    ]
    text = "\n".join(lines)
    print(text)
    (OUT_DIR / "busca2_output.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
