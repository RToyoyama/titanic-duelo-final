"""Experimento de balanceamento de classes: nenhum vs class_weight vs SMOTE."""

import pandas as pd

from titanic.config import REPORTS_DIR, TARGET
from titanic.data import load_raw
from titanic.evaluate import evaluate_pipelines
from titanic.features import build_train_test
from titanic.models import BALANCE_STRATEGIES, build_pipeline, get_model_specs


def compare_balancing(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Avalia cada modelo do catálogo com cada estratégia de balanceamento.

    Args:
        X: Features (sem o alvo).
        y: Alvo binário.

    Returns:
        DataFrame indexado por (modelo, estratégia) com acurácia, desvio, f1 e roc_auc.
    """
    pipelines = {
        f"{name}|{balance}": build_pipeline(name, balance=balance)
        for name in get_model_specs()
        for balance in BALANCE_STRATEGIES
    }
    summary, _ = evaluate_pipelines(pipelines, X, y)

    table = summary.reset_index()
    table[["model", "balance"]] = table["model"].str.split("|", expand=True)
    columns = ["acc_mean", "acc_std", "f1", "roc_auc"]
    return table.set_index(["model", "balance"])[columns].sort_index()


def main() -> None:
    """Executa o experimento e salva reports/balance_results.csv."""
    train, _ = build_train_test(*load_raw())
    X, y = train.drop(columns=TARGET), train[TARGET]

    results = compare_balancing(X, y)
    print("\n=== Acurácia média (linhas: modelo | colunas: estratégia) ===")
    print(results["acc_mean"].unstack().to_string())
    print("\n=== F1 médio ===")
    print(results["f1"].unstack().to_string())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / "balance_results.csv")


if __name__ == "__main__":
    main()
