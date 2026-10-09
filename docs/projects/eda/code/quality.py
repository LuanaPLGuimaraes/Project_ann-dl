"""Qualidade dos dados (seção 6 / item 1B do enunciado).

Cobre o que o eda.py não cobre: duplicatas ignorando o `id`, valores impossíveis,
higiene das categóricas, categorias novas no teste e a relação dos pisos com o alvo.
Tudo calculado no TREINO (o teste só entra na checagem de categorias novas).
Saída: ../output/quality_output.txt
"""
import numpy as np
import pandas as pd

from eda import CATEGORICAL_COLS, NUMERIC_COLS, OUTPUT_PATH, TARGET, Report, load_split

ID_COL = "id"
FLOOR_COLS = ["Annual_Income_USD", "Daily_Commute_km"]
QUALITY_OUTPUT = OUTPUT_PATH.parent / "quality_output.txt"
PLACEHOLDERS = {"", "unknown", "n/a", "na", "none", "null", "?", "-", "nan"}

# Regras de valor impossível. Ajuste os limites conforme a descrição do dataset no Kaggle.
RULES = {
    "Age": ("idade < 18 ou > 110", lambda s: (s < 18) | (s > 110)),
    "Annual_Income_USD": ("renda <= 0", lambda s: s <= 0),
    "Daily_Commute_km": ("deslocamento < 0 ou > 500 km", lambda s: (s < 0) | (s > 500)),
    "Number_of_Cars_Owned": ("carros < 0", lambda s: s < 0),
    "Charging_Stations_Near_Home": ("estações < 0", lambda s: s < 0),
    "Charging_Stations_Near_Work": ("estações < 0", lambda s: s < 0),
    "Environmental_Concern_Level": ("nível fora de 1-5", lambda s: s.notna() & ~s.between(1, 5)),
}
INTEGER_COLS = ["Age", "Number_of_Cars_Owned", "Charging_Stations_Near_Home",
                "Charging_Stations_Near_Work", "Environmental_Concern_Level"]


def target01(df: pd.DataFrame) -> pd.Series:
    y = df[TARGET]
    if not pd.api.types.is_numeric_dtype(y):
        y = y.map({"No": 0, "Yes": 1})
    return y.astype(float)


def missing_table(df) -> str:
    t = pd.DataFrame({"ausentes": df.isna().sum(), "%": (df.isna().mean() * 100).round(3)})
    worst = t["ausentes"].idxmax()
    if t.loc[worst, "ausentes"] == 0:
        return t.to_string() + "\n\nNenhuma coluna tem valores ausentes (0 faltantes em todas)."
    return t.to_string() + f"\n\nColuna com mais faltantes: {worst} ({t.loc[worst, 'ausentes']} linhas, {t.loc[worst, '%']}%)"


def duplicates(df) -> str:
    cols = [c for c in df.columns if c != ID_COL]
    full = int(df.duplicated().sum())
    noid = int(df.duplicated(subset=cols).sum())
    return (f"Linhas duplicadas (todas as colunas, incluindo id): {full} ({full / len(df):.2%})\n"
            f"Linhas duplicadas ignorando o id (features + alvo): {noid} ({noid / len(df):.2%})")


def column_roles(df) -> str:
    out = []
    if ID_COL in df.columns:
        out.append(f"{ID_COL}: {df[ID_COL].nunique()} valores únicos em {len(df)} linhas "
                   f"({'identificador, todos distintos' if df[ID_COL].nunique() == len(df) else 'há repetidos'})")
    const = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
    out.append(f"Colunas constantes: {const if const else 'nenhuma'}")
    return "\n".join(out)


def impossible_values(df) -> str:
    out = []
    for col, (desc, rule) in RULES.items():
        n = int(rule(df[col]).sum())
        out.append(f"{col}: {n} linhas com {desc} ({n / len(df):.2%})")
    out.append("")
    for col in INTEGER_COLS:
        n = int((df[col].dropna() % 1 != 0).sum())
        out.append(f"{col}: {n} valores não inteiros")
    return "\n".join(out)


def categorical_hygiene(train_df, test_df) -> str:
    out = []
    for col in CATEGORICAL_COLS:
        raw = train_df[col].dropna().astype(str)
        norm = raw.str.strip().str.lower()
        values = sorted(raw.unique())
        placeholders = sorted({v for v in raw.unique() if v.strip().lower() in PLACEHOLDERS})
        new_in_test = sorted(set(test_df[col].dropna().astype(str)) - set(raw))
        out.append(f"{col}: {len(values)} valores {values}")
        out.append(f"   após strip+lower: {norm.nunique()} valores "
                   f"({'sem grafias duplicadas' if norm.nunique() == len(values) else 'HÁ grafias duplicadas'})")
        out.append(f"   placeholders (Unknown, N/A, ?...): {placeholders if placeholders else 'nenhum'}")
        out.append(f"   categorias no teste ausentes do treino: {new_in_test if new_in_test else 'nenhuma'}")
    return "\n".join(out)


def floors_vs_target(df) -> str:
    y = target01(df)
    overall = y.mean()
    out = [f"Taxa geral de Yes no treino: {overall:.2%}"]
    masks = {}
    for col in FLOOR_COLS:
        floor = df[col].min()
        at = df[col] == floor
        masks[col] = at
        out.append(f"\n{col} (piso = {floor}): {int(at.sum())} linhas ({at.mean():.2%})")
        out.append(f"   taxa de Yes no piso: {y[at].mean():.2%} | fora do piso: {y[~at].mean():.2%}")
    both = masks[FLOOR_COLS[0]] & masks[FLOOR_COLS[1]]
    out.append(f"\nNos dois pisos ao mesmo tempo: {int(both.sum())} linhas ({both.mean():.2%})")
    out.append(f"Em pelo menos um piso: {int((masks[FLOOR_COLS[0]] | masks[FLOOR_COLS[1]]).sum())} linhas")
    return "\n".join(out)


def outliers_iqr(df) -> str:
    out = []
    for col in NUMERIC_COLS:
        s = df[col]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < lo) | (s > hi)).sum())
        line = f"{col}: limites [{lo:.2f}, {hi:.2f}] -> {n} outliers ({n / len(df):.2%})"
        if col in FLOOR_COLS:
            t = s[s != s.min()]
            q1b, q3b = t.quantile([0.25, 0.75])
            lob, hib = q1b - 1.5 * (q3b - q1b), q3b + 1.5 * (q3b - q1b)
            nb = int(((t < lob) | (t > hib)).sum())
            line += f"\n   sem as linhas no piso: limites [{lob:.2f}, {hib:.2f}] -> {nb} outliers ({nb / len(df):.2%} do treino)"
        out.append(line)
    return "\n".join(out)


def main():
    train_df, test_df = load_split()
    report = Report(QUALITY_OUTPUT)

    report.section("Valores ausentes (treino)")
    report.add(missing_table(train_df))
    report.section("Duplicatas")
    report.add(duplicates(train_df))
    report.section("Identificador e colunas constantes")
    report.add(column_roles(train_df))
    report.section("Valores impossíveis")
    report.add(impossible_values(train_df))
    report.section("Higiene das categóricas e categorias novas no teste")
    report.add(categorical_hygiene(train_df, test_df))
    report.section("Valores no piso x alvo")
    report.add(floors_vs_target(train_df))
    report.section("Outliers (IQR, k=1.5)")
    report.add(outliers_iqr(train_df))
    report.save()


if __name__ == "__main__":
    main()