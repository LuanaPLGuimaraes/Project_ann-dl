"""Baselines de classificação avaliados em um holdout de validação.

O split de teste definido no EDA permanece intocado. O pré-processador e cada
classificador são ajustados somente no subconjunto de treino da validação.

Uso (a partir de qualquer diretório)::

	python docs/projects/classification/code/baseline.py
	python docs/projects/classification/code/baseline.py --sem-subsidy

O resultado é gravado em ``../output/baseline_output.txt`` (ou
``baseline_output_sem-subsidy.txt``).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	balanced_accuracy_score,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
EDA_CODE_DIR = (BASE_DIR / "../../eda/code").resolve()
sys.path.insert(0, str(EDA_CODE_DIR))

from eda import RANDOM_STATE, Report, load_split  # noqa: E402
from preprocess import build_preprocessor, prepare_xy  # type: ignore[reportMissingImports]  # noqa: E402


VALIDATION_SIZE = 0.2


def build_models(use_subsidy: bool = True) -> dict[str, Pipeline]:
	"""Cria os três classificadores com o mesmo pré-processador."""
	estimators = {
		"Dummy (classe majoritária)": DummyClassifier(strategy="most_frequent"),
		"Regressão logística": LogisticRegression(max_iter=1_000),
		"Random forest": RandomForestClassifier(
			n_estimators=100,
			max_depth=20,
			min_samples_leaf=2,
			random_state=RANDOM_STATE,
			n_jobs=-1,
		),
	}
	return {
		name: Pipeline([
			("preprocess", build_preprocessor(use_subsidy=use_subsidy)),
			("classifier", estimator),
		])
		for name, estimator in estimators.items()
	}


def evaluate(model: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series) -> dict:
	"""Calcula métricas na validação, incluindo AUC a partir das probabilidades."""
	predictions = model.predict(X_valid)
	probabilities = model.predict_proba(X_valid)[:, 1]
	return {
		"Acurácia": accuracy_score(y_valid, predictions),
		"Acurácia balanceada": balanced_accuracy_score(y_valid, predictions),
		"Precisão": precision_score(y_valid, predictions, zero_division=0),
		"Revocação": recall_score(y_valid, predictions, zero_division=0),
		"F1": f1_score(y_valid, predictions, zero_division=0),
		"ROC AUC": roc_auc_score(y_valid, probabilities),
	}


def run(use_subsidy: bool = True) -> tuple[dict[str, Pipeline], pd.DataFrame]:
	"""Treina baselines e salva as métricas do holdout de validação."""
	train_df, _test_df = load_split()
	X, y = prepare_xy(train_df, use_subsidy=use_subsidy)
	X_fit, X_valid, y_fit, y_valid = train_test_split(
		X,
		y,
		test_size=VALIDATION_SIZE,
		stratify=y,
		random_state=RANDOM_STATE,
	)

	models = build_models(use_subsidy=use_subsidy)
	results = {}
	for name, model in models.items():
		model.fit(X_fit, y_fit)
		results[name] = evaluate(model, X_valid, y_valid)

	metrics = pd.DataFrame.from_dict(results, orient="index")
	metrics.index.name = "Modelo"
	metrics = metrics * 100

	suffix = "" if use_subsidy else "_sem-subsidy"
	output_path = BASE_DIR / f"../output/baseline_output{suffix}.txt"
	report = Report(output_path.resolve())
	report.section(f"Baselines de classificação (use_subsidy={use_subsidy})")
	report.add(
		"Split oficial de teste não utilizado. Validação estratificada retirada "
		f"do treino: ajuste={len(X_fit):,} linhas; validação={len(X_valid):,} linhas."
	)
	report.add(
		"Distribuição na validação: "
		+ y_valid.value_counts(normalize=True).sort_index().mul(100).round(2).to_dict().__str__()
		+ " (0=No, 1=Yes)."
	)
	report.add("Threshold de classificação: 0,5; nenhum balanceamento de classes aplicado.")
	report.add("Métricas de validação (%):")
	report.add(metrics.round(2).to_string())
	report.save()
	return models, metrics


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--sem-subsidy",
		action="store_true",
		help="exclui Subsidy_Available, conforme a comparação planejada no EDA",
	)
	args = parser.parse_args()
	run(use_subsidy=not args.sem_subsidy)
