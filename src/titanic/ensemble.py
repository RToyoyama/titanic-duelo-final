"""Ensembles dos modelos tunados: diagnóstico de diversidade, votação e stacking."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from titanic.config import RANDOM_STATE, REPORTS_DIR, TARGET
from titanic.data import load_raw
from titanic.evaluate import evaluate_pipelines, paired_comparison
from titanic.features import build_train_test
from titanic.models import build_pipeline
from titanic.tuning import load_best_params

BestParams = dict[str, dict[str, object]]


def build_base_estimators(
    best_params: BestParams, probability: bool = False
) -> list[tuple[str, BaseEstimator]]:
    """Monta os pipelines tunados que compõem os ensembles.

    Args:
        best_params: Hiperparâmetros por modelo (saída do tuning).
        probability: Calibra a SVM para expor `predict_proba` (voto soft e stacking).
            Substitui `SVC(probability=True)`, depreciado no scikit-learn 1.9.

    Returns:
        Lista de tuplas (nome, pipeline não ajustado).
    """
    estimators = []
    for name, params in best_params.items():
        pipeline = build_pipeline(name, params=params)
        if name == "svm" and probability:
            svc = pipeline.named_steps["model"]
            pipeline.set_params(model=CalibratedClassifierCV(svc, ensemble=False))
        estimators.append((name, pipeline))
    return estimators


def build_ensembles(best_params: BestParams) -> dict[str, BaseEstimator]:
    """Cria os ensembles candidatos.

    Args:
        best_params: Hiperparâmetros por modelo (saída do tuning).

    Returns:
        Dicionário nome -> ensemble não ajustado.
    """
    return {
        "voting_hard": VotingClassifier(build_base_estimators(best_params), voting="hard"),
        "voting_soft": VotingClassifier(
            build_base_estimators(best_params, probability=True), voting="soft"
        ),
        "stacking": StackingClassifier(
            build_base_estimators(best_params, probability=True),
            final_estimator=LogisticRegression(max_iter=1000),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            stack_method="predict_proba",
        ),
    }


def oof_predictions(
    estimators: list[tuple[str, BaseEstimator]], X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    """Gera previsões out-of-fold (cada linha prevista por um modelo que não a viu).

    Args:
        estimators: Lista de tuplas (nome, pipeline).
        X: Features (sem o alvo).
        y: Alvo binário.

    Returns:
        DataFrame com uma coluna de previsões por modelo.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return pd.DataFrame(
        {name: cross_val_predict(est, X, y, cv=cv, n_jobs=-1) for name, est in estimators}
    )


def error_matrix(oof: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Marca onde cada modelo errou.

    Args:
        oof: Previsões out-of-fold por modelo.
        y: Alvo verdadeiro.

    Returns:
        DataFrame com 1 onde o modelo errou e 0 onde acertou.
    """
    return oof.apply(lambda col: col.to_numpy() != y.to_numpy()).astype(int)


def main() -> None:
    """Diagnostica a diversidade e compara os ensembles com o melhor modelo individual."""
    train, _ = build_train_test(*load_raw())
    X, y = train.drop(columns=TARGET), train[TARGET]
    best = load_best_params()

    errors = error_matrix(oof_predictions(build_base_estimators(best), X, y), y)
    print("\n=== Acurácia OOF por modelo ===")
    print((1 - errors.mean()).round(4).to_string())
    print("\n=== Correlação dos erros (menor = mais diversidade) ===")
    print(errors.corr().round(3).to_string())
    print(f"\nOráculo (ao menos um acerta): {(errors == 0).any(axis=1).mean():.4f}")
    print(f"Todos erram: {int(errors.all(axis=1).sum())} passageiros")

    pipelines = {
        "random_forest": build_pipeline("random_forest", params=best["random_forest"]),
        **build_ensembles(best),
    }
    summary, folds = evaluate_pipelines(pipelines, X, y)
    print("\n=== Duelo final (acc 5x3 CV) ===")
    print(summary[["acc_mean", "acc_std", "roc_auc", "f1"]].to_string())
    print("\n=== Comparação pareada vs random_forest ===")
    print(paired_comparison(folds, reference="random_forest").to_string())

    summary.to_csv(REPORTS_DIR / "ensemble_results.csv")


if __name__ == "__main__":
    main()
