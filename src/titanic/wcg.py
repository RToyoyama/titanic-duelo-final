"""Regra woman-child-group (WCG) aplicada sobre as previsões de um classificador."""

from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted

from titanic.config import REPORTS_DIR, TARGET
from titanic.data import load_raw
from titanic.ensemble import build_ensembles
from titanic.evaluate import evaluate_pipelines, paired_comparison
from titanic.features import build_train_test
from titanic.models import build_pipeline
from titanic.tuning import load_best_params

WCG_GROUPS: dict[str, str] = {"wcg": "FamilyKey", "wcg_ticket": "Ticket"}


class WomanChildGroupOverride(ClassifierMixin, BaseEstimator):
    """Ajusta as previsões de um classificador com a regra woman-child-group.

    A regra só age em grupos com mulheres/crianças de rótulo conhecido no treino:
        - Menino (Master) cujo grupo teve todas as mulheres/crianças vivas -> 1.
        - Mulher cujo grupo teve todas as mulheres/crianças mortas -> 0.

    As taxas dos grupos vêm apenas do `fit` (fold de treino), sem vazamento na CV.

    Args:
        estimator: Classificador base (não ajustado).
        group_col: Coluna que define o grupo (FamilyKey ou Ticket).
    """

    def __init__(self, estimator: BaseEstimator, group_col: str = "FamilyKey") -> None:
        self.estimator = estimator
        self.group_col = group_col

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> Self:
        """Ajusta o modelo base e calcula a taxa de sobrevivência WCG por grupo.

        Args:
            X: Features com `group_col`, IsWomanChild, Title e Sex.
            y: Alvo binário.

        Returns:
            O próprio classificador ajustado.
        """
        y_arr = np.asarray(y)
        self.estimator_ = clone(self.estimator).fit(X, y_arr)
        self.classes_ = self.estimator_.classes_

        is_wc = X["IsWomanChild"].to_numpy() == 1
        groups = X.loc[is_wc, self.group_col].to_numpy()
        self.group_rate_ = pd.Series(y_arr[is_wc]).groupby(groups).mean().to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prevê com o modelo base e aplica a regra WCG.

        Args:
            X: Features com `group_col`, Title e Sex.

        Returns:
            Array de previsões (0/1).
        """
        check_is_fitted(self, "group_rate_")
        preds = np.asarray(self.estimator_.predict(X)).copy()
        rate = X[self.group_col].map(self.group_rate_).to_numpy(dtype=float)

        is_boy = (X["Title"] == "Master").to_numpy()
        is_female = (X["Sex"] == "female").to_numpy()
        preds[is_boy & (rate == 1)] = 1
        preds[is_female & (rate == 0)] = 0
        return preds


def main() -> None:
    """Compara os modelos finais com e sem a regra WCG (CV e impacto no teste)."""
    train, test = build_train_test(*load_raw())
    X, y = train.drop(columns=TARGET), train[TARGET]
    best = load_best_params()

    bases = {
        "random_forest": build_pipeline("random_forest", params=best["random_forest"]),
        "voting_hard": build_ensembles(best)["voting_hard"],
    }
    pipelines: dict[str, BaseEstimator] = {}
    for name, base in bases.items():
        pipelines[name] = base
        for suffix, column in WCG_GROUPS.items():
            pipelines[f"{name}+{suffix}"] = WomanChildGroupOverride(base, group_col=column)

    summary, folds = evaluate_pipelines(pipelines, X, y)
    print("\n=== Acurácia (5x3 CV) ===")
    print(summary[["acc_mean", "acc_std", "f1"]].to_string())

    for name in bases:
        subset = folds[folds["model"].str.startswith(name)]
        print(f"\n=== Pareada vs {name} ===")
        print(paired_comparison(subset, reference=name).to_string())

    print("\n=== Previsões alteradas pela regra no teste (treino completo) ===")
    for name, base in bases.items():
        base_preds = clone(base).fit(X, y).predict(test)
        for suffix, column in WCG_GROUPS.items():
            wcg = WomanChildGroupOverride(base, group_col=column).fit(X, y)
            changed = int((wcg.predict(test) != base_preds).sum())
            print(f"{name}+{suffix}: {changed} passageiros")

    summary.to_csv(REPORTS_DIR / "wcg_results.csv")


if __name__ == "__main__":
    main()
