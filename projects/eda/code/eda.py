import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "../../../../data/train.csv"
FIGURES_DIR = BASE_DIR / "../figures"

RANDOM_STATE = 42
TARGET = "Will_Buy_EV"


# split de train e test
def load_split(test_size: float = 0.2):
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# plot da distribuição do alvo - Figura 1
def plot_target_distribution(train_df: pd.DataFrame):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    counts = train_df[TARGET].value_counts(normalize=True).sort_index()

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values, color=["#2a78d6", "#eb6834"])
    ax.set_ylabel("Proporção")
    ax.set_title(f"Distribuição de {TARGET} (treino)")
    ax.set_ylim(0, 1)

    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.1%}",
            ha="center",
        )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig01-distribuicao-alvo.svg")
    plt.close(fig)


NUMERIC_COLS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

# descrição numérica do dataset 
def describe_numeric(train_df: pd.DataFrame):
    return train_df[NUMERIC_COLS].describe().T.round(2)
    

# features delimitadas como relevantes para plotar no histograma
HIST_FEATURES = ["Age", "Annual_Income_USD", "Daily_Commute_km"]
def plot_univariate_histograms(train_df: pd.DataFrame):
    for i, col in enumerate(HIST_FEATURES, start=2):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(train_df[col], bins=50, color="#2a78d6")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequência")
        ax.set_title(f"Distribuição de {col} (treino)")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()

        fig_name = f"fig{i:02d}-{col.lower().replace('_', '-')}.svg"
        fig.savefig(FIGURES_DIR / fig_name)
        plt.close(fig)
        print(f"figura salva em {FIGURES_DIR / fig_name}")

# analisando o acumulo de dados no valor mínimo de algumas features numéricas
def check_floor_spike(train_df: pd.DataFrame, cols: list[str]):
    for col in cols:
        min_val = train_df[col].min()
        count_at_min = (train_df[col] == min_val).sum()
        pct = count_at_min / len(train_df) * 100
        print(f"{col}: {count_at_min} amostras ({pct:.2f}%) exatamente no valor mínimo ({min_val})")

if __name__ == "__main__":
    train_df, test_df = load_split()

    print("=== Split ===")
    print("treino:", train_df.shape)
    print("teste:", test_df.shape)

    print("\n=== Distribuição do alvo (treino) ===")
    print(train_df[TARGET].value_counts(normalize=True))

    print("\n=== Tipos (dtypes) ===")
    print(train_df.dtypes)

    print("\n=== Cardinalidade (nunique) ===")
    print(train_df.nunique())

    print("\n=== Estatísticas descritivas (numéricas) ===")
    print(describe_numeric(train_df))

    print("\n=== Categorias (colunas nominais) ===")
    for col in ["Gender", "City_Type", "Current_Car_Type"]:
        print(f"{col}: {train_df[col].unique()}")

    plot_target_distribution(train_df)
    print(f"\nFigura salva em: {FIGURES_DIR / 'fig01-distribuicao-alvo.svg'}")

    plot_univariate_histograms(train_df)
    check_floor_spike(train_df, ["Annual_Income_USD", "Daily_Commute_km"])