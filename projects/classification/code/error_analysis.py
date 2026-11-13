"""Etapa 6: análise de erros.

Subgrupos e piores exemplos usam as previsões JÁ salvas da etapa 5 (não reavaliam o teste).
A importância por permutação roda na VALIDAÇÃO, com o modelo final (seed 42).

    python docs/projects/classification/code/error_analysis.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from data_prep import (  # noqa: E402
    RANDOM_STATE,
    VALIDATION_SIZE,
    build_preprocessor,
    load_split,
    prepare_xy,
)
from network import MLP  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR.parent / "output"
FIG_DIR = BASE_DIR.parent / "figures"
MIN_N = 1000          # subgrupos menores que isso não entram no ranking
N_REPEATS = 5         # repetições da permutação
N_WORST = 300         # nº de piores exemplos usados para ver o padrão

CAT_VARS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
            "Subsidy_Available", "Number_of_Cars_Owned", "Environmental_Concern_Level"]
Q_VARS = ["Age", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work"]
FLOOR_VARS = ["Annual_Income_USD", "Daily_Commute_km"]


def masks_cat(s):
    s = s.round().astype("Int64").astype(str) if s.dtype.kind == "f" else s.astype(str)
    return [(v, (s == v).to_numpy()) for v in sorted(s.unique())]


def masks_q(s, q=4, floor=False):
    out = []
    rest = pd.Series(True, index=s.index)
    if floor:
        f = s.min()
        out.append((f"piso ({f:g})", (s == f).to_numpy()))
        rest = s != f
    codes = pd.qcut(s[rest], q, duplicates="drop")
    for iv in codes.cat.categories:
        m = pd.Series(False, index=s.index)
        m.loc[codes.index[(codes == iv).to_numpy()]] = True
        out.append((str(iv), m.to_numpy()))
    return out


def summarize(mask, y, pred, c):
    yy, pp = y[mask], pred[mask]
    n = int(mask.sum())
    pos = int(yy.sum())
    neg = n - pos
    fn = int(((pp == 0) & (yy == 1)).sum())
    fp = int(((pp == 1) & (yy == 0)).sum())
    tp = pos - fn
    return {
        "n": n,
        "% Yes real": pos / n * 100,
        "recall": tp / pos * 100 if pos else np.nan,
        "FPR": fp / neg * 100 if neg else np.nan,
        "precisão": tp / (tp + fp) * 100 if (tp + fp) else np.nan,
        "erro %": (fp + fn) / n * 100,
        "custo": (fp + c * fn) / n,
    }


def table(rows):
    return pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}")


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(np.asarray(X, dtype=np.float32)))).numpy()


def main():
    # ---- dados: mesma divisão da etapa 5 -------------------------------------
    train_df, test_df = load_split()
    X_all, y_all = prepare_xy(train_df)
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_all, y_all, test_size=VALIDATION_SIZE, stratify=y_all, random_state=RANDOM_STATE)
    cutoff = float(y_fit.mean())
    c = (1 - cutoff) / cutoff

    X_test, y_test = prepare_xy(test_df)
    X_test = X_test.reset_index(drop=True)
    saved = np.load(OUT_DIR / "etapa5_probs_teste.npz")
    y = saved["y"].astype(int)
    p = saved["Rede_s42"].astype(float)
    assert len(y) == len(X_test) and (y_test.to_numpy() == y).all(), "ordem do teste não bate com as previsões salvas"
    pred = (p >= cutoff).astype(int)

    L = []
    L.append(f"Previsões da rede (seed 42) no TESTE, corte {cutoff:.4f}, custo FN/FP = {c:.2f}")
    L.append("Colunas: recall, FPR (FP entre os 'No'), precisão, erro % = (FP+FN)/n, custo = (FP + c*FN)/n.")
    L.append("")
    all_rows = [{"grupo": "GERAL", **summarize(np.ones(len(y), bool), y, pred, c)}]
    L.append(table(all_rows))

    # ---- erro por classe real e por faixa de probabilidade --------------------
    L.append("\n== Erro por classe real (faixas do alvo) ==")
    L.append(table([{"grupo": f"real = {name}", **summarize(y == v, y, pred, c)} for v, name in ((0, "No"), (1, "Yes"))]))

    L.append("\n== Calibração: probabilidade prevista x fração real de 'Yes' ==")
    edges = [0, 0.05, 0.10, cutoff, 0.30, 0.50, 0.70, 0.90, 1.0001]
    cal = (pd.DataFrame({"p": p, "y": y, "faixa": pd.cut(p, edges, right=False)})
           .groupby("faixa", observed=True)
           .agg(n=("y", "size"), prob_media=("p", "mean"), pct_yes_real=("y", "mean")))
    cal[["prob_media", "pct_yes_real"]] *= 100
    L.append(cal.to_string(float_format=lambda v: f"{v:.2f}"))

    # ---- erro por subgrupo ----------------------------------------------------
    ranking = []
    for var in CAT_VARS + Q_VARS + FLOOR_VARS:
        s = X_test[var]
        if var in CAT_VARS:
            groups = masks_cat(s)
        else:
            groups = masks_q(s, floor=var in FLOOR_VARS)
        rows = [{"grupo": g, **summarize(m, y, pred, c)} for g, m in groups]
        L.append(f"\n== {var} ==")
        L.append(table(rows))
        ranking += [{"variável": var, **r} for r in rows if r["n"] >= MIN_N]

    rk = pd.DataFrame(ranking)
    L.append(f"\n== Ranking de subgrupos (n >= {MIN_N}) por erro % ==")
    L.append(rk.sort_values("erro %", ascending=False).head(8).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    L.append(f"\n== Ranking de subgrupos (n >= {MIN_N}) por custo ==")
    L.append(rk.sort_values("custo", ascending=False).head(8).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # ---- piores exemplos ------------------------------------------------------
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    feats = list(X_test.columns)
    pos_idx, neg_idx = np.where(y == 1)[0], np.where(y == 0)[0]
    fn_order = pos_idx[np.argsort(p[pos_idx])]            # compradores com menor probabilidade
    fp_order = neg_idx[np.argsort(-p[neg_idx])]           # não compradores com maior probabilidade
    for name, order in (("FALSOS NEGATIVOS mais graves (comprou, rede deu prob. baixíssima)", fn_order),
                        ("FALSOS POSITIVOS mais graves (não comprou, rede deu prob. altíssima)", fp_order)):
        top = X_test.iloc[order[:10]].copy()
        top.insert(0, "prob", p[order[:10]].round(4))
        L.append(f"\n== {name} ==")
        L.append(top.to_string())

    for name, order, ref in (("FN", fn_order[:N_WORST], pos_idx), ("FP", fp_order[:N_WORST], neg_idx)):
        worst, base = X_test.iloc[order], X_test.iloc[ref]
        cls = "compradores (Yes)" if name == "FN" else "não compradores (No)"
        L.append(f"\n== Padrão dos {N_WORST} piores {name}: % no grupo dos piores vs. % em todos os {cls} ==")
        rows = []
        for var in ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible", "Subsidy_Available"]:
            for v in sorted(base[var].astype(str).unique()):
                a = (worst[var].astype(str) == v).mean() * 100
                b = (base[var].astype(str) == v).mean() * 100
                rows.append({"variável": var, "valor": v, "% piores": a, "% classe": b, "razão": a / b if b else np.nan})
        L.append(table(rows))
        rows = []
        for var in ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
                    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Environmental_Concern_Level"]:
            rows.append({"variável": var, "mediana piores": worst[var].median(), "mediana classe": base[var].median()})
        for var in FLOOR_VARS:
            f = X_test[var].min()
            rows.append({"variável": f"{var} no piso (%)", "mediana piores": (worst[var] == f).mean() * 100,
                         "mediana classe": (base[var] == f).mean() * 100})
        L.append(table(rows))

    # ---- importância por permutação (VALIDAÇÃO) ------------------------------
    pre = build_preprocessor(use_subsidy=True)
    Xf_t = pre.fit_transform(X_fit)
    model = MLP(Xf_t.shape[1], hidden_dims=(64, 32), dropout=0.1)
    model.load_state_dict(torch.load(OUT_DIR / "modelo_final_seed42.pt"))
    yv = y_valid.to_numpy()

    def pr_auc(Xraw):
        return average_precision_score(yv, predict(model, pre.transform(Xraw))) * 100

    base_score = pr_auc(X_valid)
    L.append(f"\n== Importância por permutação (VALIDAÇÃO, rede seed 42) ==")
    L.append(f"PR-AUC de referência na validação: {base_score:.2f} (esperado ~73,89)")
    L.append(f"Queda da PR-AUC (pontos) ao embaralhar cada variável, média ± desvio de {N_REPEATS} repetições:")
    rng = np.random.default_rng(RANDOM_STATE)
    imp = []
    for var in X_valid.columns:
        drops = []
        for _ in range(N_REPEATS):
            Xp = X_valid.copy()
            Xp[var] = rng.permutation(Xp[var].to_numpy())
            drops.append(base_score - pr_auc(Xp))
        imp.append((var, float(np.mean(drops)), float(np.std(drops, ddof=1))))
    imp.sort(key=lambda t: -t[1])
    for var, m, s in imp:
        L.append(f"  {var:<30}{m:7.2f} ± {s:.2f}")

    text = "\n".join(L)
    print(text)
    (OUT_DIR / "analise_erros_output.txt").write_text(text + "\n", encoding="utf-8")

    # ---- figuras --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    names = [t[0] for t in imp][::-1]
    ax.barh(names, [t[1] for t in imp][::-1], xerr=[t[2] for t in imp][::-1], color="#1f77b4")
    ax.set_xlabel("Queda da PR-AUC na validação (pontos)")
    ax.set_title("Importância por permutação (rede, seed 42)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig07-importancia-permutacao.svg")
    plt.close(fig)

    top = rk.sort_values("erro %", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([f"{r['variável']}: {r['grupo']}" for _, r in top.iterrows()], top["erro %"], color="#d62728")
    ax.axvline(all_rows[0]["erro %"], ls="--", color="gray", label=f"Geral ({all_rows[0]['erro %']:.1f}%)")
    ax.set_xlabel("Erro (% de exemplos errados)")
    ax.set_title("Subgrupos com maior erro no teste (rede, corte 0,1746)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig06-erro-por-subgrupo.svg")
    plt.close(fig)
    print("\nSaídas: output/analise_erros_output.txt, figures/fig06 e fig07")


if __name__ == "__main__":
    main()