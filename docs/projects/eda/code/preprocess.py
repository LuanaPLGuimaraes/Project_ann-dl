"""Pré-processamento (etapa 4A + 4C).

Importável:
    from preprocess import build_preprocessor, prepare_xy
    X_train, y_train = prepare_xy(train_df, use_subsidy=True)
    pre = build_preprocessor(use_subsidy=True)
    X_train_t = pre.fit_transform(X_train)   # fit só no treino
    X_test_t = pre.transform(X_test)         # teste apenas transformado

Decisões vindas do EDA bivariado:
  - Range_Anxiety_Level é DESCARTADA (derivada do alvo -> vazamento).
  - Subsidy_Available é testada COM e SEM (use_subsidy=True/False): prevê o alvo
    quase sozinha, mas não é derivada dele, então pode ser um sinal legítimo.

Rodando como script, grava o relatório em ../output/preprocess_output.txt
(ou preprocess_output_sem-subsidy.txt com --sem-subsidy).
"""
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from eda import OUTPUT_PATH, TARGET, Report, load_split

ID_COL = "id"  # identificador: descartado

# Derivada do alvo (vazamento): nunca entra no modelo.
LEAK_COLS = ["Range_Anxiety_Level"]

# Forte preditora, mas não derivada do alvo: testada com e sem.
SUBSIDY_COL = "Subsidy_Available"

# Features com pico exato no valor mínimo (9,20% e 21,59% no treino).
# Hipótese: não-resposta disfarçada -> o piso vira NaN, é imputado e ganha um indicador.
FLOOR_COLS = ["Annual_Income_USD", "Daily_Commute_km"]

# Numéricas sem tratamento especial (só imputação de segurança + escala)
PLAIN_NUMERIC = [
    "Age",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

# Nominais / binárias (sem Subsidy_Available, que entra condicionalmente)
NOMINAL_COLS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
]


def nominal_cols(use_subsidy: bool = True) -> list[str]:
    return NOMINAL_COLS + ([SUBSIDY_COL] if use_subsidy else [])


def output_path(use_subsidy: bool = True):
    suffix = "" if use_subsidy else "_sem-subsidy"
    return OUTPUT_PATH.parent / f"preprocess_output{suffix}.txt"


class FloorToNaN(BaseEstimator, TransformerMixin):
    """Aprende o valor mínimo de cada coluna NO TREINO e o troca por NaN."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.floors_ = np.nanmin(X, axis=0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        X[X == self.floors_] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class IQRClipper(BaseEstimator, TransformerMixin):
    """Winsorização por IQR (limites aprendidos só no treino, ignorando NaN)."""

    def __init__(self, k: float = 1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def n_rows_affected(self, X) -> int:
        X = np.asarray(X, dtype=float)
        return int(((X < self.lower_) | (X > self.upper_)).any(axis=1).sum())

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def prepare_xy(df: pd.DataFrame, use_subsidy: bool = True):
    """Separa X e y; descarta `id` e as colunas de vazamento; mapeia o alvo Yes/No para 1/0."""
    y = df[TARGET]
    if y.dtype == object or str(y.dtype).startswith("str"):
        y = y.map({"No": 0, "Yes": 1})
    drop = [TARGET, ID_COL] + LEAK_COLS + ([] if use_subsidy else [SUBSIDY_COL])
    X = df.drop(columns=drop, errors="ignore")
    return X, y.astype(int)


def build_preprocessor(use_subsidy: bool = True) -> ColumnTransformer:
    floor_pipe = Pipeline([
        ("floor", FloorToNaN()),
        ("clip", IQRClipper()),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    plain_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    nominal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore", drop="if_binary", sparse_output=False
        )),
    ])
    return ColumnTransformer(
        [
            ("floor_num", floor_pipe, FLOOR_COLS),
            ("plain_num", plain_pipe, PLAIN_NUMERIC),
            ("nominal", nominal_pipe, nominal_cols(use_subsidy)),
        ],
        verbose_feature_names_out=False,
    )


def describe_preprocessing(pre, X_train, X_train_t, X_test_t) -> str:
    """Texto com os números que o relatório precisa citar."""
    floor_pipe = pre.named_transformers_["floor_num"]
    floor_step = floor_pipe.named_steps["floor"]
    clip_step = floor_pipe.named_steps["clip"]
    X_floor = floor_step.transform(X_train[FLOOR_COLS])

    lines = []
    lines.append("Valores no piso (viram NaN e são imputados pela mediana do treino):")
    for col, floor, n in zip(FLOOR_COLS, floor_step.floors_, np.isnan(X_floor).sum(axis=0)):
        lines.append(f"  {col}: piso={floor} -> {n} linhas ({n / len(X_train):.2%})")

    lines.append("\nLimites IQR (k=1.5) aprendidos no treino, já sem os pisos:")
    for col, lo, hi in zip(FLOOR_COLS, clip_step.lower_, clip_step.upper_):
        lines.append(f"  {col}: [{lo:.2f}, {hi:.2f}]")
    n_out = clip_step.n_rows_affected(X_floor)
    lines.append(f"Linhas com algum outlier winsorizado: {n_out} ({n_out / len(X_train):.2%})")

    lines.append(f"\nColunas fora do modelo: {ID_COL} (identificador), {', '.join(LEAK_COLS)} (derivada do alvo)")
    lines.append(f"Subsidy_Available no modelo: {'Subsidy_Available' in X_train.columns}")
    lines.append("\nFaixa das numéricas ANTES do escalonamento (justifica o StandardScaler):")
    ranges = X_train[FLOOR_COLS + PLAIN_NUMERIC].agg(["min", "max"]).T
    lines.append(ranges.to_string())

    names = pre.get_feature_names_out()
    lines.append(f"\nShape treino após pipeline: {X_train_t.shape}")
    lines.append(f"Shape teste após pipeline:  {X_test_t.shape}")
    lines.append(f"NaN no treino: {int(np.isnan(X_train_t).sum())} | NaN no teste: {int(np.isnan(X_test_t).sum())}")
    lines.append(f"\nNomes das {len(names)} features:")
    lines.extend(f"  {i:02d}. {n}" for i, n in enumerate(names, start=1))
    return "\n".join(lines)


def run(use_subsidy: bool = True):
    train_df, test_df = load_split()
    X_train, y_train = prepare_xy(train_df, use_subsidy)
    X_test, y_test = prepare_xy(test_df, use_subsidy)

    pre = build_preprocessor(use_subsidy)
    X_train_t = pre.fit_transform(X_train)   # fit só no treino
    X_test_t = pre.transform(X_test)         # teste apenas transformado

    report = Report(output_path(use_subsidy))
    report.section(f"Pipeline de pré-processamento (use_subsidy={use_subsidy})")
    report.add(describe_preprocessing(pre, X_train, X_train_t, X_test_t))
    report.save()
    return pre, X_train_t, X_test_t, y_train, y_test


if __name__ == "__main__":
    run(use_subsidy="--sem-subsidy" not in sys.argv)