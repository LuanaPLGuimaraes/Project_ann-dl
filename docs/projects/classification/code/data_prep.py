"""Dados para a rede: divisão treino/validação e tensores PyTorch.

Usa exatamente a mesma validação das baselines (20% do treino, estratificada,
seed 42). O teste oficial não é usado aqui por padrão, ele só entra na etapa 5
(include_test=True, chamado apenas por evaluate_test.py).

    python docs/projects/classification/code/data_prep.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
EDA_CODE_DIR = (BASE_DIR / "../../eda/code").resolve()
sys.path.insert(0, str(EDA_CODE_DIR))

from eda import RANDOM_STATE, load_split  # noqa: E402
from preprocess import build_preprocessor, prepare_xy  # type: ignore[reportMissingImports]  # noqa: E402

VALIDATION_SIZE = 0.2


def load_tensors(use_subsidy: bool = True, include_test: bool = False) -> dict:
    """Devolve X/y de ajuste e de validação como tensores float32.

    Com include_test=True devolve também X_test/y_test (só a etapa 5 usa).
    """
    train_df, test_df = load_split()
    X, y = prepare_xy(train_df, use_subsidy=use_subsidy)
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(use_subsidy=use_subsidy)
    X_fit_t = preprocessor.fit_transform(X_fit)  # aprende só com o ajuste
    X_valid_t = preprocessor.transform(X_valid)

    def to_x(array) -> torch.Tensor:
        return torch.tensor(np.asarray(array, dtype=np.float32))

    def to_y(series) -> torch.Tensor:
        return torch.tensor(series.to_numpy(dtype=np.float32))

    data = {
        "X_fit": to_x(X_fit_t),
        "y_fit": to_y(y_fit),
        "X_valid": to_x(X_valid_t),
        "y_valid": to_y(y_valid),
        "n_inputs": X_fit_t.shape[1],
    }

    if include_test:  # só a etapa 5 chega aqui
        X_test, y_test = prepare_xy(test_df, use_subsidy=use_subsidy)
        data["X_test"] = to_x(preprocessor.transform(X_test))
        data["y_test"] = to_y(y_test)

    return data


if __name__ == "__main__":
    d = load_tensors()
    print("X_fit:", tuple(d["X_fit"].shape), "| y_fit:", tuple(d["y_fit"].shape))
    print("X_valid:", tuple(d["X_valid"].shape), "| y_valid:", tuple(d["y_valid"].shape))
    print("Colunas de entrada:", d["n_inputs"])
    print(f"% de 'Yes' no ajuste: {d['y_fit'].mean().item() * 100:.2f}")
    print(f"% de 'Yes' na validação: {d['y_valid'].mean().item() * 100:.2f}")
    print("Há NaN?", bool(torch.isnan(d["X_fit"]).any() or torch.isnan(d["X_valid"]).any()))