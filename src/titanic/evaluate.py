"""Duelo de modelos: comparação com validação cruzada estratificada repetida."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from titanic.config import FIGURES_DIR, RANDOM_STATE, REPORTS_DIR, TARGET
from titanic.data import load_raw
from titanic.features import build_train_test
from titanic.models import build_pipeline, get_model_specs

SCORING: dict[str, str] = {"accuracy": "accuracy", "roc_auc": "roc_auc", "f1": "f1"}


def make_cv(n_splits: int = 5, n_repeats: int = 3) -> RepeatedStratifiedKFold:
    """Cria o esquema de CV padrão do projeto (mesmos folds para todos os modelos).

    Args:
        n_splits: Número de folds.
        n_repeats: Número de repetições com embaralhamentos diferentes.

    Returns:
        Objeto de validação cruzada com seed fixa.
    """
    return RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE
    )


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    model_names: list[str] | None = None,
    cv: RepeatedStratifiedKFold | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Avalia os modelos do catálogo com validação cruzada.

    Args:
        X: Features (saída de `build_train_test`, sem o alvo).
        y: Alvo binário.
        model_names: Modelos a avaliar; todos do catálogo se None.
        cv: Esquema de CV; `make_cv()` se None.

    Returns:
        Tupla (resumo por modelo, scores por fold).
    """
    cv = make_cv() if cv is None else cv
    names = model_names or list(get_model_specs())

    folds = []
    for name in names:
        res = cross_validate(
            build_pipeline(name), X, y, cv=cv, scoring=SCORING, return_train_score=True, n_jobs=-1
        )
        folds.append(
            pd.DataFrame(
                {
                    "model": name,
                    "fold": range(len(res["test_accuracy"])),
                    "accuracy": res["test_accuracy"],
                    "roc_auc": res["test_roc_auc"],
                    "f1": res["test_f1"],
                    "train_accuracy": res["train_accuracy"],
                    "fit_time": res["fit_time"],
                }
            )
        )
    fold_scores = pd.concat(folds, ignore_index=True)

    summary = (
        fold_scores.groupby("model")
        .agg(
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            roc_auc=("roc_auc", "mean"),
            f1=("f1", "mean"),
            train_acc=("train_accuracy", "mean"),
            fit_time_s=("fit_time", "mean"),
        )
        .assign(overfit_gap=lambda d: d["train_acc"] - d["acc_mean"])
        .sort_values("acc_mean", ascending=False)
        .round(4)
    )
    return summary, fold_scores


def paired_comparison(fold_scores: pd.DataFrame, reference: str = "logreg") -> pd.DataFrame:
    """Compara cada modelo com a referência fold a fold (mesmos splits).

    Args:
        fold_scores: Scores por fold (saída de `evaluate_models`).
        reference: Modelo usado como base de comparação.

    Returns:
        DataFrame com diferença média, desvio da diferença e taxa de vitórias.
    """
    pivot = fold_scores.pivot(index="fold", columns="model", values="accuracy")
    diff = pivot.sub(pivot[reference], axis=0).drop(columns=reference)
    return (
        pd.DataFrame(
            {
                "mean_diff": diff.mean(),
                "std_diff": diff.std(),
                "win_rate": (diff > 0).mean(),
            }
        )
        .sort_values("mean_diff", ascending=False)
        .round(4)
    )


def plot_duel(fold_scores: pd.DataFrame, name: str = "baseline_duel") -> None:
    """Salva um boxplot da acurácia por modelo em reports/figures.

    Args:
        fold_scores: Scores por fold.
        name: Nome do arquivo (sem extensão).
    """
    order = fold_scores.groupby("model")["accuracy"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=fold_scores, x="model", y="accuracy", order=order, ax=ax)
    ax.set_title("Duelo baseline — acurácia por fold (5x3 CV)")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Executa o duelo baseline e salva resultados e gráfico."""
    train, _ = build_train_test(*load_raw())
    X, y = train.drop(columns=TARGET), train[TARGET]

    summary, fold_scores = evaluate_models(X, y)
    print("\n=== Resumo (ordenado por acurácia) ===")
    print(summary.to_string())
    print("\n=== Comparação pareada vs logreg ===")
    print(paired_comparison(fold_scores).to_string())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_DIR / "baseline_results.csv")
    plot_duel(fold_scores)


if __name__ == "__main__":
    main()
