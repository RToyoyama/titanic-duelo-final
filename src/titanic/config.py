"""Configurações centrais do projeto: caminhos, seed e nomes de colunas."""

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
SUBMISSIONS_DIR: Final[Path] = DATA_DIR / "submissions"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"

COMPETITION: Final[str] = "titanic"
TARGET: Final[str] = "Survived"
ID_COL: Final[str] = "PassengerId"
RANDOM_STATE: Final[int] = 42
