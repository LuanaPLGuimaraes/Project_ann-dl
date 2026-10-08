import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (BASE_DIR / "../../data/train.csv").resolve()
FIGURES_DIR = (BASE_DIR / "../figures").resolve()
OUTPUT_PATH = (BASE_DIR / "../output/eda_output.txt").resolve()

RANDOM_STATE = 42
TARGET = "Will_Buy_EV"

NUMERIC_COLS = [
    "Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
CATEGORICAL_COLS = [
    "Gender", "City_Type", "Current_Car_Type",
    "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level",
]
HIST_FEATURES = ["Age", "Annual_Income_USD", "Daily_Commute_km"]


class Report:
    """Acumula o texto do EDA e grava tudo em um .txt no final."""

    def __init__(self, path: Path):
        self.path = path
        self.lines: list[str] = []

    def section(self, title: str):
        if self.lines:
            self.lines.append("")
        self.lines.append(f"=== {title} ===")

    def add(self, text=""):
        self.lines.append(str(text))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def load_split(test_size: float = 0.2):
    df = pd.read_csv(DATA_PATH)
    target_mapping = {"No": 0, "Yes": 1}
    df[TARGET] = df[TARGET].map(target_mapping)
    if df[TARGET].isna().any():
        unknown = sorted(df.loc[df[TARGET].isna(), TARGET].unique())
        raise ValueError(f"Valores desconhecidos em {TARGET}: {unknown}")
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_fig(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_target_distribution(train_df, fig_name="fig01-distribuicao-alvo.svg"):
    counts = train_df[TARGET].value_counts(normalize=True).sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index.astype(str), counts.values, color=["#2a78d6", "#eb6834"])
    ax.set_ylabel("Proporção")
    ax.set_title(f"Distribuição de {TARGET} (treino)")
    ax.set_ylim(0, 1)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in counts.values], padding=3)
    return [save_fig(fig, fig_name)]


def plot_univariate_histograms(train_df, start: int = 2):
    saved = []
    for i, col in enumerate(HIST_FEATURES, start=start):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(train_df[col].dropna(), bins=50, color="#2a78d6")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequência")
        ax.set_title(f"Distribuição de {col} (treino)")
        saved.append(save_fig(fig, f"fig{i:02d}-{col.lower().replace('_', '-')}.svg"))
    return saved


def plot_numeric_by_target(train_df, start: int = 5):
    """Histograma sobreposto por classe do alvo."""
    saved = []
    for i, col in enumerate(HIST_FEATURES, start=start):
        fig, ax = plt.subplots(figsize=(6, 4))
        for cls, group in train_df.groupby(TARGET):
            ax.hist(group[col].dropna(), bins=40, alpha=0.5, label=str(cls), density=True)
        ax.set_xlabel(col)
        ax.set_ylabel("Densidade")
        ax.set_title(f"{col} por {TARGET}")
        ax.legend(title=TARGET)
        saved.append(save_fig(fig, f"fig{i:02d}-{col.lower().replace('_', '-')}-por-alvo.svg"))
    return saved


def plot_categorical_vs_target(train_df, start: int = 8):
    """Taxa média do alvo por categoria (alvo precisa ser 0/1)."""
    saved = []
    for i, col in enumerate(CATEGORICAL_COLS, start=start):
        rate = train_df.groupby(col)[TARGET].mean().sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(rate.index.astype(str), rate.values, color="#2a78d6")
        ax.axvline(train_df[TARGET].mean(), color="gray", linestyle="--", label="média geral")
        ax.set_xlabel(f"Taxa de {TARGET}")
        ax.set_title(f"{TARGET} por {col}")
        ax.legend()
        saved.append(save_fig(fig, f"fig{i:02d}-{col.lower().replace('_', '-')}-por-alvo.svg"))
    return saved


def plot_correlation(train_df, fig_name):
    corr = train_df[NUMERIC_COLS + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlação (treino)")
    return [save_fig(fig, fig_name)]


def describe_numeric(train_df):
    return train_df[NUMERIC_COLS].describe().T.round(2)


def check_missing_and_duplicates(train_df) -> str:
    missing = train_df.isna().sum()
    missing_txt = missing[missing > 0].to_string() if missing.any() else "sem valores nulos"
    return f"{missing_txt}\nlinhas duplicadas: {train_df.duplicated().sum()}"


def check_floor_spike(train_df, cols) -> str:
    out = []
    for col in cols:
        min_val = train_df[col].min()
        n = (train_df[col] == min_val).sum()
        out.append(f"{col}: {n} amostras ({n / len(train_df):.2%}) no valor mínimo ({min_val})")
    return "\n".join(out)


def check_outliers_iqr(train_df, cols) -> str:
    out = []
    for col in cols:
        q1, q3 = train_df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (train_df[col] < q1 - 1.5 * iqr) | (train_df[col] > q3 + 1.5 * iqr)
        out.append(f"{col}: {mask.sum()} outliers ({mask.mean():.2%})")
    return "\n".join(out)


if __name__ == "__main__":
    report = Report(OUTPUT_PATH)
    train_df, test_df = load_split()

    report.section("Split")
    report.add(f"treino: {train_df.shape} | teste: {test_df.shape}")

    report.section("Distribuição do alvo (treino)")
    report.add(train_df[TARGET].value_counts(normalize=True).to_string())

    report.section("Tipos e cardinalidade")
    report.add(pd.DataFrame({"dtype": train_df.dtypes, "nunique": train_df.nunique()}).to_string())

    report.section("Nulos e duplicatas")
    report.add(check_missing_and_duplicates(train_df))

    report.section("Estatísticas descritivas (numéricas)")
    report.add(describe_numeric(train_df).to_string())

    report.section("Categorias (frequência relativa)")
    for col in CATEGORICAL_COLS:
        report.add(f"\n{col}:")
        report.add(train_df[col].value_counts(normalize=True, dropna=False).round(3).to_string())

    report.section("Outliers (IQR)")
    report.add(check_outliers_iqr(train_df, HIST_FEATURES))

    report.section("Acúmulo no valor mínimo")
    report.add(check_floor_spike(train_df, ["Annual_Income_USD", "Daily_Commute_km"]))

    figures = []
    figures += plot_target_distribution(train_df)
    figures += plot_univariate_histograms(train_df, start=2)
    figures += plot_numeric_by_target(train_df, start=5)
    figures += plot_categorical_vs_target(train_df, start=8)
    figures += plot_correlation(train_df, "fig14-correlacao.svg")

    report.section("Figuras salvas")
    for path in figures:
        report.add(path)

    report.save()