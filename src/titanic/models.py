"""Catálogo de modelos candidatos e montagem dos pipelines completos."""

from dataclasses import dataclass
from typing import Literal

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from titanic.config import RANDOM_STATE
from titanic.preprocessing import build_preprocessor

BalanceStrategy = Literal["none", "class_weight", "smote"]
BALANCE_STRATEGIES: tuple[BalanceStrategy, ...] = ("none", "class_weight", "smote")

# Razão negativos/positivos no treino (549/342): equivale a class_weight="balanced" no XGBoost.
XGB_POS_WEIGHT: float = 549 / 342


@dataclass(frozen=True)
class ModelSpec:
    """Especificação de um modelo candidato.

    Attributes:
        name: Identificador curto do modelo.
        estimator: Classificador não ajustado.
        needs_scaling: Se o modelo é sensível à escala das features.
    """

    name: str
    estimator: ClassifierMixin
    needs_scaling: bool


def get_model_specs() -> dict[str, ModelSpec]:
    """Retorna o catálogo de modelos com instâncias novas a cada chamada.

    Returns:
        Dicionário nome -> ModelSpec.
    """
    specs = [
        ModelSpec("logreg", LogisticRegression(max_iter=1000), needs_scaling=True),
        ModelSpec("svm", SVC(kernel="rbf", random_state=RANDOM_STATE), needs_scaling=True),
        ModelSpec(
            "random_forest",
            RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
            needs_scaling=False,
        ),
        ModelSpec(
            "xgboost",
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
            ),
            needs_scaling=False,
        ),
    ]
    return {spec.name: spec for spec in specs}


def _class_weight_params(name: str) -> dict[str, object]:
    if name == "xgboost":
        return {"model__scale_pos_weight": XGB_POS_WEIGHT}
    return {"model__class_weight": "balanced"}


def build_pipeline(
    name: str,
    params: dict[str, object] | None = None,
    balance: BalanceStrategy = "none",
    group_survival: bool = True,
) -> Pipeline:
    """Monta o pipeline completo (pré-processamento + balanceamento + modelo).

    Usa o Pipeline do imbalanced-learn: o SMOTE roda apenas no `fit`
    (dentro de cada fold), nunca no `predict`.

    Args:
        name: Nome do modelo no catálogo.
        params: Hiperparâmetros no formato do Pipeline (ex.: {"model__C": 1.0}).
        balance: Estratégia de balanceamento de classes.
        group_survival: Inclui a feature de sobrevivência do grupo.

    Returns:
        Pipeline não ajustado.

    Raises:
        KeyError: Se o modelo não existir no catálogo.
        ValueError: Se a estratégia de balanceamento for inválida.
    """
    specs = get_model_specs()
    if name not in specs:
        raise KeyError(f"Modelo desconhecido: {name!r}. Opções: {sorted(specs)}")
    if balance not in BALANCE_STRATEGIES:
        raise ValueError(f"Balanceamento inválido: {balance!r}. Opções: {BALANCE_STRATEGIES}")

    spec = specs[name]
    preprocessor = build_preprocessor(scale=spec.needs_scaling, group_survival=group_survival)
    steps: list[tuple[str, object]] = [("prep", preprocessor)]
    if balance == "smote":
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", spec.estimator))

    pipeline = Pipeline(steps)
    if params:
        pipeline.set_params(**params)
    if balance == "class_weight":
        pipeline.set_params(**_class_weight_params(name))
    return pipeline
