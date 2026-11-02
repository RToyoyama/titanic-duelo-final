"""Tuning de hiperparâmetros com GridSearchCV e RandomizedSearchCV."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold

from titanic.config import BEST_PARAMS_PATH, RANDOM_STATE, REPORTS_DIR, TARGET
from titanic.data import load_raw
from titanic.evaluate import evaluate_pipelines
from titanic.features import build_train_test
from titanic.models import build_pipeline


@dataclass(frozen=True)
class SearchSpec:
    """Espaço de busca de um modelo.

    Attributes:
        method: "grid" (exaustiva) ou "random" (amostragem).
        space: Parâmetros no formato do Pipeline (ex.: "model__C").
        n_iter: Combinações sorteadas (apenas para "random").
    """

    method: Literal["grid", "random"]
    space: dict[str, object] = field(default_factory=dict)
    n_iter: int = 0


SEARCH_SPECS: dict[str, SearchSpec] = {
    "logreg": SearchSpec("grid", {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}),
    "svm": SearchSpec(
        "grid",
        {"model__C": [0.3, 1.0, 3.0, 10.0], "model__gamma": ["scale", 0.01, 0.03, 0.1]},
    ),
    "random_forest": SearchSpec(
        "random",
        {
            "model__n_estimators": randint(200, 800),
            "model__max_depth": [4, 5, 6, 7, 8, None],
            "model__min_samples_leaf": randint(1, 10),
            "model__max_features": ["sqrt", 0.5, 0.7],
        },
        n_iter=40,
    ),
    "xgboost": SearchSpec(
        "random",
        {
            "model__n_estimators": randint(100, 600),
            "model__max_depth": randint(2, 6),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.5, 0.5),
            "model__min_child_weight": randint(1, 8),
            "model__reg_lambda": loguniform(0.1, 10),
        },
        n_iter=40,
    ),
}


def make_search_cv() -> RepeatedStratifiedKFold:
    """CV da busca, com seed diferente da avaliação final (reduz viés de seleção).

    Returns:
        Esquema 5x2 com seed RANDOM_STATE + 1.
    """
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE + 1)


def to_jsonable(params: dict[str, object]) -> dict[str, object]:
    """Converte escalares numpy (np.int64, np.float64) em tipos nativos do Python.

    Args:
        params: Hiperparâmetros vindos do `best_params_`.

    Returns:
        Dicionário serializável em JSON.
    """
    return {key: value.item() if hasattr(value, "item") else value for key, value in params.items()}


def run_search(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv: RepeatedStratifiedKFold | None = None,
) -> tuple[dict[str, object], float]:
    """Executa a busca de hiperparâmetros de um modelo.

    Args:
        name: Modelo do catálogo (chave de SEARCH_SPECS).
        X: Features (sem o alvo).
        y: Alvo binário.
        cv: Esquema de CV; `make_search_cv()` se None.

    Returns:
        Tupla (melhores parâmetros, acurácia média da busca).
    """
    spec = SEARCH_SPECS[name]
    cv = make_search_cv() if cv is None else cv
    common = {"scoring": "accuracy", "cv": cv, "n_jobs": -1, "refit": False}

    if spec.method == "grid":
        search = GridSearchCV(build_pipeline(name), spec.space, **common)
    else:
        search = RandomizedSearchCV(
            build_pipeline(name),
            spec.space,
            n_iter=spec.n_iter,
            random_state=RANDOM_STATE,
            **common,
        )
    search.fit(X, y)
    return to_jsonable(search.best_params_), float(search.best_score_)


def save_best_params(best: dict[str, dict[str, object]], path: Path = BEST_PARAMS_PATH) -> None:
    """Salva os melhores hiperparâmetros em JSON.

    Args:
        best: Dicionário modelo -> parâmetros.
        path: Caminho do arquivo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(best, indent=2), encoding="utf-8")


def load_best_params(path: Path = BEST_PARAMS_PATH) -> dict[str, dict[str, object]]:
    """Carrega os melhores hiperparâmetros salvos.

    Args:
        path: Caminho do arquivo.

    Returns:
        Dicionário modelo -> parâmetros.

    Raises:
        FileNotFoundError: Se o tuning ainda não foi executado.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado. Rode `make tune` primeiro.")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Executa o tuning, salva os parâmetros e compara tuned vs baseline."""
    train, _ = build_train_test(*load_raw())
    X, y = train.drop(columns=TARGET), train[TARGET]

    best_params: dict[str, dict[str, object]] = {}
    search_scores: dict[str, float] = {}
    for name in SEARCH_SPECS:
        print(f"Tuning {name}...", flush=True)
        best_params[name], search_scores[name] = run_search(name, X, y)
    save_best_params(best_params)

    pipelines = {}
    for name in SEARCH_SPECS:
        pipelines[f"{name}|baseline"] = build_pipeline(name)
        pipelines[f"{name}|tuned"] = build_pipeline(name, params=best_params[name])
    _, folds = evaluate_pipelines(pipelines, X, y)

    acc = folds.pivot(index="fold", columns="model", values="accuracy")
    results = pd.DataFrame(
        {
            name: {
                "search_score": search_scores[name],
                "baseline": acc[f"{name}|baseline"].mean(),
                "tuned": acc[f"{name}|tuned"].mean(),
                "tuned_std": acc[f"{name}|tuned"].std(),
                "win_rate": (acc[f"{name}|tuned"] > acc[f"{name}|baseline"]).mean(),
            }
            for name in SEARCH_SPECS
        }
    ).T
    results["gain"] = results["tuned"] - results["baseline"]
    results = results.sort_values("tuned", ascending=False).round(4)

    print("\n=== Tuned vs baseline (acc 5x3 CV, seed de avaliação) ===")
    print(results.to_string())
    print("\n=== Melhores parâmetros ===")
    print(json.dumps(best_params, indent=2))
    results.to_csv(REPORTS_DIR / "tuning_results.csv")


if __name__ == "__main__":
    main()
