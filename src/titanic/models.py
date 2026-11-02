"""Catálogo de modelos candidatos e montagem dos pipelines completos."""

from dataclasses import dataclass

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from titanic.config import RANDOM_STATE
from titanic.preprocessing import build_preprocessor


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


def build_pipeline(name: str, group_survival: bool = True) -> Pipeline:
    """Monta o pipeline completo (pré-processamento + modelo).

    Args:
        name: Nome do modelo no catálogo.
        group_survival: Inclui a feature de sobrevivência do grupo.

    Returns:
        Pipeline não ajustado.

    Raises:
        KeyError: Se o modelo não existir no catálogo.
    """
    specs = get_model_specs()
    if name not in specs:
        raise KeyError(f"Modelo desconhecido: {name!r}. Opções: {sorted(specs)}")

    spec = specs[name]
    preprocessor = build_preprocessor(scale=spec.needs_scaling, group_survival=group_survival)
    return Pipeline([("prep", preprocessor), ("model", spec.estimator)])
