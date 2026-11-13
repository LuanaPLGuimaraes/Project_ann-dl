"""Etapa 5: avaliação final no TESTE (uso único).

    python docs/projects/classification/code/evaluate_test.py --ensaio   # na validação, 1 seed
    python docs/projects/classification/code/evaluate_test.py            # teste real (uma vez)
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

from data_prep import load_tensors  # noqa: E402
from train import fit  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR.parent / "output"
FIG_DIR = BASE_DIR.parent / "figures"
SEEDS = (42, 43, 44)
CHOSEN = dict(hidden_dims=(64, 32), dropout=0.1, weight_decay=1e-2)  # config escolhida (etapa 4)
NAMES = ["Rede", "Regressão logística", "Random forest", "Boosting (ref. extra)"]
BASELINES = ("Regressão logística", "Random forest")  # critério da etapa 0 vale contra estes


def predict_net(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X)).numpy()


def metrics_at(y, p, cutoff, cost_ratio):
    pred = (p >= cutoff).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "PR-AUC": average_precision_score(y, p) * 100,
        "ROC-AUC": roc_auc_score(y, p) * 100,
        "F1 macro": f1.mean() * 100,
        "Precisão (Yes)": prec[1] * 100, "Recall (Yes)": rec[1] * 100, "F1 (Yes)": f1[1] * 100,
        "Precisão (No)": prec[0] * 100, "Recall (No)": rec[0] * 100, "F1 (No)": f1[0] * 100,
        "Custo/exemplo": (fp + cost_ratio * fn) / len(y),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def ms(values):
    v = np.asarray(values, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def fmt(values, nd=2):
    m, s = ms(values)
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def col(results, name, key):
    return [r[key] for r in results[name]]


def plot_pr(y, p42, cutoff, path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, p in p42.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p) * 100
        ax.plot(rec[::25], prec[::25], lw=2.2 if name == "Rede" else 1.2, label=f"{name} (PR-AUC {ap:.1f})")
    pred = (p42["Rede"] >= cutoff).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    ax.scatter([tp / y.sum()], [tp / max(pred.sum(), 1)], color="black", zorder=5,
               label=f"Corte {cutoff:.4f} (rede)")
    ax.axhline(y.mean(), ls="--", color="gray", label=f"Acaso ({y.mean() * 100:.1f}%)")
    ax.set_xlabel("Recall (Yes)")
    ax.set_ylabel("Precisão (Yes)")
    ax.set_title("Curva precisão-recall")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cm(cm, path):
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({norm[i, j] * 100:.1f}%)".replace(",", "."),
                    ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1], ["No", "Yes"])
    ax.set_yticks([0, 1], ["No", "Yes"])
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusão (rede, seed 42)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensaio", action="store_true", help="roda na validação (1 seed), sem tocar no teste")
    parser.add_argument("--forcar", action="store_true", help="permite repetir a avaliação no teste")
    args = parser.parse_args()
    ensaio = args.ensaio
    seeds = (42,) if ensaio else SEEDS

    if ensaio:
        out_dir = fig_dir = Path(tempfile.mkdtemp(prefix="ensaio_etapa5_"))
        report_path = out_dir / "ensaio_etapa5_output.txt"
    else:
        out_dir, fig_dir = OUT_DIR, FIG_DIR
        report_path = out_dir / "etapa5_teste_output.txt"
        if report_path.exists() and not args.forcar:
            sys.exit("O teste JÁ foi avaliado (etapa5_teste_output.txt existe). Não repita: o teste só pode ser usado uma vez.")
    out_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    t0 = time.time()
    data = load_tensors(include_test=not ensaio)
    if ensaio:
        X_eval, y_eval, label = data["X_valid"], data["y_valid"], "VALIDAÇÃO (ENSAIO, não são resultados do teste)"
    else:
        X_eval, y_eval, label = data["X_test"], data["y_test"], "TESTE (uso único)"
    y = y_eval.numpy().astype(int)
    cutoff = data["y_fit"].mean().item()  # prevalência do ajuste, definido antes de ver o teste
    cost_ratio = (1 - cutoff) / cutoff
    Xf, yf, Xe = data["X_fit"].numpy(), data["y_fit"].numpy(), X_eval.numpy()

    probs = {n: {} for n in NAMES}
    best_model = None
    for seed in seeds:
        print(f"[seed {seed}] treinando rede, random forest e boosting...", flush=True)
        model, _ = fit(data, seed=seed, verbose=False, **CHOSEN)
        probs["Rede"][seed] = predict_net(model, X_eval)
        if seed == 42:
            best_model = model
        rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2,
                                    random_state=seed, n_jobs=-1).fit(Xf, yf)
        probs["Random forest"][seed] = rf.predict_proba(Xe)[:, 1]
        hgb = HistGradientBoostingClassifier(random_state=seed).fit(Xf, yf)
        probs["Boosting (ref. extra)"][seed] = hgb.predict_proba(Xe)[:, 1]
    print("treinando regressão logística...", flush=True)
    lr = LogisticRegression(max_iter=1000).fit(Xf, yf)
    probs["Regressão logística"][42] = lr.predict_proba(Xe)[:, 1]  # determinística: 1 execução

    results = {n: [metrics_at(y, p, cutoff, cost_ratio) for p in by_seed.values()] for n, by_seed in probs.items()}

    L = []
    L.append(f"Avaliação: {label}")
    L.append(f"Exemplos: {len(y)} | % Yes: {y.mean() * 100:.2f} | seeds da rede: {list(seeds)}")
    L.append(f"Corte (prevalência do ajuste): {cutoff:.4f} | razão de custo implícita FN/FP: {cost_ratio:.2f}")
    L.append("Valores em %, média ± desvio-padrão amostral entre seeds (n=1: desvio 0).")
    L.append("")
    cols = ["PR-AUC", "ROC-AUC", "F1 macro", "Precisão (Yes)", "Recall (Yes)", "F1 (Yes)"]
    L.append("Tabela comparativa (mesmo conjunto, mesmo corte)")
    L.append(f"{'Modelo':<24}" + "".join(f"{c:>17}" for c in cols))
    for n in NAMES:
        L.append(f"{n:<24}" + "".join(f"{fmt(col(results, n, c)):>17}" for c in cols))
    L.append(f"{'Acaso (PR-AUC)':<24}{y.mean() * 100:>17.2f}")
    L.append("")
    L.append("Rede, por classe (corte escolhido)")
    L.append(f"{'Classe':<8}{'Precisão':>16}{'Recall':>16}{'F1':>16}")
    for cls in ("No", "Yes"):
        L.append(f"{cls:<8}" + "".join(f"{fmt(col(results, 'Rede', f'{m} ({cls})')):>16}" for m in ("Precisão", "Recall", "F1")))
    L.append("")
    cm42 = np.array([[results["Rede"][0]["tn"], results["Rede"][0]["fp"]],
                     [results["Rede"][0]["fn"], results["Rede"][0]["tp"]]])
    L.append("Matriz de confusão da rede (seed 42): linhas = real (No, Yes), colunas = previsto (No, Yes)")
    L.append(str(cm42))
    L.append("")
    L.append(f"Custo médio por exemplo (FP = 1, FN = {cost_ratio:.2f}); menor é melhor")
    for n in NAMES:
        L.append(f"  {n:<24}{fmt(col(results, n, 'Custo/exemplo'), 3)}")
    L.append(f"  {'Contactar todos':<24}{(y == 0).mean():.3f}")
    L.append(f"  {'Não contactar ninguém':<24}{cost_ratio * (y == 1).mean():.3f}")
    L.append("")
    net = col(results, "Rede", "PR-AUC")
    best_name = max(BASELINES, key=lambda n: ms(col(results, n, "PR-AUC"))[0])
    best_mean = ms(col(results, best_name, "PR-AUC"))[0]
    diff = ms(net)[0] - best_mean
    std_net = ms(net)[1]
    hgb_diff = ms(net)[0] - ms(col(results, "Boosting (ref. extra)", "PR-AUC"))[0]
    L.append("Critério da etapa 0 (rede ≥ melhor baseline + 1 ponto de PR-AUC e diferença > desvio entre seeds)")
    L.append(f"  Melhor baseline (LogReg/RF): {best_name} = {best_mean:.2f}")
    L.append(f"  Rede - melhor baseline = {diff:+.2f} p.p. | desvio da rede entre seeds = {std_net:.2f}")
    L.append(f"  Diferença maior que a variação entre seeds? {'SIM' if abs(diff) > std_net else 'NÃO'}")
    L.append(f"  Critério atingido? {'SIM' if (diff >= 1.0 and diff > std_net) else 'NÃO'}")
    L.append(f"  Rede - boosting (referência extra) = {hgb_diff:+.2f} p.p.")
    L.append(f"\nTempo total: {time.time() - t0:.0f} s")

    text = "\n".join(L)
    print("\n" + text)
    report_path.write_text(text + "\n", encoding="utf-8")

    p42 = {n: probs[n][42] for n in NAMES}
    plot_pr(y, p42, cutoff, fig_dir / ("fig04-curva-pr-teste.svg"))
    plot_cm(cm42, fig_dir / "fig05-matriz-confusao-teste.svg")

    if not ensaio:
        arrays = {"y": y}
        for n in NAMES:
            for s, p in probs[n].items():
                arrays[f"{n}_s{s}"] = p.astype(np.float32)
        np.savez_compressed(out_dir / "etapa5_probs_teste.npz", **arrays)
        torch.save(best_model.state_dict(), out_dir / "modelo_final_seed42.pt")
    print(f"\nSaídas em: {out_dir if ensaio else 'output/ e figures/'}")


if __name__ == "__main__":
    main()