"""Feature engineering determinística (não usa o alvo).

As features deste módulo não dependem de `Survived`, então podem ser calculadas
sobre treino + teste concatenados sem vazamento de alvo. Features que aprendem
estatísticas dos dados ficam em `titanic.transformers` (ajustadas dentro da CV).
"""

import numpy as np
import pandas as pd

from titanic.config import PROCESSED_DIR, TARGET
from titanic.data import load_raw

COMMON_TITLES: tuple[str, ...] = ("Mr", "Mrs", "Miss", "Master")
TITLE_ALIASES: dict[str, str] = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
DECK_GROUPS: dict[str, str] = {
    **dict.fromkeys("ABCT", "ABC"),
    **dict.fromkeys("DE", "DE"),
    **dict.fromkeys("FG", "FG"),
}


def extract_title(names: pd.Series) -> pd.Series:
    """Extrai o pronome de tratamento do nome, agrupando títulos raros.

    Args:
        names: Série no formato "Sobrenome, Título. Nome".

    Returns:
        Série com um de: Mr, Mrs, Miss, Master ou Rare.
    """
    titles = names.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    titles = titles.replace(TITLE_ALIASES)
    return titles.where(titles.isin(COMMON_TITLES), "Rare")


def family_band(family_size: pd.Series) -> pd.Series:
    """Agrupa o tamanho da família em faixas (relação em U observada na EDA).

    Args:
        family_size: Tamanho da família (SibSp + Parch + 1).

    Returns:
        Série com Alone (1), Small (2-4) ou Large (5+).
    """
    bands = pd.cut(family_size, bins=[0, 1, 4, np.inf], labels=["Alone", "Small", "Large"])
    return bands.astype(str)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria as features determinísticas.

    `TicketGroupSize` conta passageiros de todo o DataFrame recebido; passe
    treino + teste concatenados para contar grupos completos (uso transdutivo,
    válido no Kaggle, mas não reproduzível em produção).

    Args:
        df: Dados brutos (camada raw).

    Returns:
        Cópia de `df` com as novas colunas.
    """
    out = df.copy()
    out["Title"] = extract_title(out["Name"])
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["FamilyBand"] = family_band(out["FamilySize"])
    out["HasCabin"] = out["Cabin"].notna().astype(int)
    out["Deck"] = out["Cabin"].str[0].map(DECK_GROUPS).fillna("U")
    out["TicketGroupSize"] = out.groupby("Ticket")["Ticket"].transform("size")
    out["FarePerPerson"] = out["Fare"] / out["TicketGroupSize"]
    out["Surname"] = out["Name"].str.split(",").str[0].str.strip()
    out["FamilyKey"] = out["Surname"] + "_" + out["Ticket"].astype(str).str[:-1]
    out["IsWomanChild"] = ((out["Sex"] == "female") | (out["Title"] == "Master")).astype(int)
    return out


def build_train_test(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica `build_features` em treino + teste juntos e separa de volta.

    Args:
        train: Treino bruto (com alvo).
        test: Teste bruto (sem alvo).

    Returns:
        Tupla (train_features, test_features).
    """
    full = build_features(pd.concat([train, test], ignore_index=True))
    n_train = len(train)

    train_feat = full.iloc[:n_train].reset_index(drop=True)
    train_feat[TARGET] = train_feat[TARGET].astype(int)
    test_feat = full.iloc[n_train:].drop(columns=TARGET).reset_index(drop=True)
    return train_feat, test_feat


def main() -> None:
    """Gera a camada processed a partir da camada raw."""
    train_feat, test_feat = build_train_test(*load_raw())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_feat.to_csv(PROCESSED_DIR / "train_features.csv", index=False)
    test_feat.to_csv(PROCESSED_DIR / "test_features.csv", index=False)
    print(f"Salvo em {PROCESSED_DIR} | train {train_feat.shape} | test {test_feat.shape}")


if __name__ == "__main__":
    main()
