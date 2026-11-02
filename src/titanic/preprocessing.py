"""Pré-processamento: ColumnTransformer com imputação, transformação, escala e encoding."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from titanic.transformers import GroupMedianImputer, GroupSurvivalEncoder

NUMERIC_COLS: list[str] = ["Pclass", "SibSp", "Parch", "FamilySize", "TicketGroupSize", "HasCabin"]
FARE_COLS: list[str] = ["FarePerPerson"]
AGE_COLS: list[str] = ["Title", "Age"]
CATEGORICAL_COLS: list[str] = ["Sex", "Embarked", "Title", "FamilyBand", "Deck"]
GROUP_COLS: list[str] = ["Ticket"]


def _scaler(scale: bool) -> StandardScaler | str:
    return StandardScaler() if scale else "passthrough"


def _numeric_pipeline(scale: bool, log: bool = False) -> Pipeline:
    steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if log:
        steps.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    steps.append(("scale", _scaler(scale)))
    return Pipeline(steps)


def build_preprocessor(scale: bool = True, group_survival: bool = True) -> ColumnTransformer:
    """Monta o pré-processador completo do projeto.

    Args:
        scale: Padroniza as numéricas (essencial para SVM/LogReg, indiferente para árvores).
        group_survival: Inclui a feature de sobrevivência do grupo de ticket (H6 da EDA).

    Returns:
        ColumnTransformer não ajustado, com saída em pandas DataFrame.
    """
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    age = Pipeline([("impute", GroupMedianImputer()), ("scale", _scaler(scale))])

    transformers: list[tuple[str, object, list[str]]] = [
        ("num", _numeric_pipeline(scale), NUMERIC_COLS),
        ("fare", _numeric_pipeline(scale, log=True), FARE_COLS),
        ("age", age, AGE_COLS),
        ("cat", categorical, CATEGORICAL_COLS),
    ]
    if group_survival:
        transformers.append(("group", GroupSurvivalEncoder(), GROUP_COLS))

    preprocessor = ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    )
    return preprocessor.set_output(transform="pandas")
