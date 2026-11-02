"""Transformers customizados compatíveis com scikit-learn.

Ficam aqui as features que aprendem estatísticas dos dados e, por isso,
precisam ser ajustadas dentro de cada fold da validação cruzada.
"""

from collections.abc import Sequence
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class GroupMedianImputer(TransformerMixin, BaseEstimator):
    """Imputa uma coluna numérica pela mediana do seu grupo.

    Grupos sem nenhum valor conhecido recebem a mediana global.

    Args:
        group_col: Coluna que define os grupos (ex.: Title).
        value_col: Coluna a imputar (ex.: Age).
    """

    def __init__(self, group_col: str = "Title", value_col: str = "Age") -> None:
        self.group_col = group_col
        self.value_col = value_col

    def fit(self, X: pd.DataFrame, y: ArrayLike | None = None) -> Self:
        """Aprende as medianas por grupo e a mediana global.

        Args:
            X: DataFrame com `group_col` e `value_col`.
            y: Ignorado; existe por compatibilidade com o Pipeline.

        Returns:
            O próprio imputer ajustado.
        """
        medians = X.groupby(self.group_col)[self.value_col].median().dropna()
        self.medians_ = medians.to_dict()
        self.global_median_ = float(X[self.value_col].median())
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Preenche os faltantes de `value_col`.

        Args:
            X: DataFrame com `group_col` e `value_col`.

        Returns:
            Array (n, 1) com a coluna imputada.
        """
        check_is_fitted(self, "medians_")
        fill = X[self.group_col].map(self.medians_).fillna(self.global_median_)
        return X[self.value_col].fillna(fill).to_numpy(dtype=float).reshape(-1, 1)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Nome da coluna de saída."""
        return np.array([self.value_col], dtype=object)


class GroupSurvivalEncoder(TransformerMixin, BaseEstimator):
    """Taxa de sobrevivência dos *outros* membros do grupo (ex.: mesmo ticket).

    Estratégia anti-vazamento:
        - `fit_transform` (treino): leave-one-out — o rótulo da própria linha
          nunca entra no cálculo da sua feature.
        - `transform` (validação/teste): usa todos os membros vistos no fit.

    Dentro de um Pipeline, o scikit-learn chama `fit_transform` no fold de
    treino e `transform` no fold de validação, mantendo a CV honesta.

    Args:
        default: Valor para quem não tem outros membros conhecidos no grupo.
    """

    def __init__(self, default: float = 0.5) -> None:
        self.default = default

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> Self:
        """Calcula soma e contagem do alvo por grupo.

        Args:
            X: DataFrame com uma única coluna de chave de grupo.
            y: Alvo binário.

        Returns:
            O próprio encoder ajustado.
        """
        stats = (
            pd.DataFrame({"key": self._keys(X), "y": np.asarray(y, dtype=float)})
            .groupby("key")["y"]
            .agg(["sum", "count"])
        )
        self.sum_ = stats["sum"].to_dict()
        self.count_ = stats["count"].to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Aplica a taxa do grupo inteiro (dados não vistos no fit).

        Args:
            X: DataFrame com uma única coluna de chave de grupo.

        Returns:
            Array (n, 1) com a taxa de sobrevivência do grupo.
        """
        check_is_fitted(self, "sum_")
        keys = self._keys(X)
        return self._finalize(keys.map(self.sum_) / keys.map(self.count_))

    def fit_transform(self, X: pd.DataFrame, y: ArrayLike | None = None) -> np.ndarray:
        """Ajusta e transforma o treino com leave-one-out.

        Args:
            X: DataFrame com uma única coluna de chave de grupo.
            y: Alvo binário (obrigatório).

        Returns:
            Array (n, 1) com a taxa de sobrevivência dos outros membros.

        Raises:
            ValueError: Se `y` não for informado.
        """
        if y is None:
            raise ValueError("GroupSurvivalEncoder requer y no fit_transform.")
        self.fit(X, y)
        keys = self._keys(X)
        others_sum = keys.map(self.sum_) - np.asarray(y, dtype=float)
        others_count = keys.map(self.count_) - 1
        return self._finalize(others_sum / others_count.replace(0, np.nan))

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Nome da coluna de saída."""
        return np.array(["group_survival"], dtype=object)

    def _finalize(self, rate: pd.Series) -> np.ndarray:
        return rate.fillna(self.default).to_numpy(dtype=float).reshape(-1, 1)

    @staticmethod
    def _keys(X: pd.DataFrame) -> pd.Series:
        return pd.Series(np.asarray(X).ravel()).astype(str)
