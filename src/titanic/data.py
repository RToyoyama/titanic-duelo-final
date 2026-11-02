"""Leitura e validação dos dados brutos (camada raw)."""

from pathlib import Path

import pandas as pd

from titanic.config import ID_COL, RAW_DIR, TARGET

FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        ID_COL,
        "Pclass",
        "Name",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Ticket",
        "Fare",
        "Cabin",
        "Embarked",
    }
)
TRAIN_COLUMNS: frozenset[str] = FEATURE_COLUMNS | {TARGET}


def load_csv(path: Path, required_columns: frozenset[str]) -> pd.DataFrame:
    """Lê um CSV e valida a presença das colunas obrigatórias.

    Args:
        path: Caminho do arquivo CSV.
        required_columns: Colunas que devem existir no arquivo.

    Returns:
        DataFrame com o conteúdo do arquivo.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se faltar alguma coluna obrigatória.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado. Rode `make data` primeiro.")

    df = pd.read_csv(path)
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} sem colunas obrigatórias: {sorted(missing)}")
    return df


def load_raw(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os conjuntos de treino e teste da camada raw.

    Args:
        raw_dir: Diretório com train.csv e test.csv.

    Returns:
        Tupla (train, test).
    """
    train = load_csv(raw_dir / "train.csv", TRAIN_COLUMNS)
    test = load_csv(raw_dir / "test.csv", FEATURE_COLUMNS)
    return train, test
