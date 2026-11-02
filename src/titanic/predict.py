"""Treino final com todo o conjunto de treino e geração das submissões do Kaggle."""

import argparse
from itertools import combinations

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

from titanic.config import ID_COL, MODELS_DIR, RAW_DIR, SUBMISSIONS_DIR, TARGET
from titanic.data import load_raw
from titanic.ensemble import build_ensembles
from titanic.features import build_train_test
from titanic.models import build_pipeline, get_model_specs
from titanic.tuning import load_best_params

DEFAULT_MODELS: tuple[str, ...] = ("voting_hard", "random_forest")


def get_final_estimator(name: str, best_params: dict[str, dict[str, object]]) -> BaseEstimator:
    """Retorna um modelo do catálogo ou um ensemble, já com os parâmetros tunados.

    Args:
        name: Nome de um modelo do catálogo ou de um ensemble.
        best_params: Hiperparâmetros por modelo (saída do tuning).

    Returns:
        Estimador não ajustado.

    Raises:
        KeyError: Se o nome não corresponder a nenhum modelo ou ensemble.
    """
    if name in get_model_specs():
        return build_pipeline(name, params=best_params[name])

    ensembles = build_ensembles(best_params)
    if name in ensembles:
        return ensembles[name]

    options = sorted(get_model_specs()) + sorted(ensembles)
    raise KeyError(f"Modelo final desconhecido: {name!r}. Opções: {options}")


def make_submission(model: BaseEstimator, test: pd.DataFrame) -> pd.DataFrame:
    """Gera o DataFrame de submissão no formato exigido pelo Kaggle.

    Args:
        model: Estimador ajustado.
        test: Features do teste (sem o alvo).

    Returns:
        DataFrame com PassengerId e Survived (inteiros).
    """
    return pd.DataFrame({ID_COL: test[ID_COL].astype(int), TARGET: model.predict(test).astype(int)})


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame) -> None:
    """Valida o formato da submissão antes do envio.

    Args:
        submission: DataFrame de submissão.
        test: Features do teste usadas na predição.

    Raises:
        ValueError: Se colunas, número de linhas, IDs ou valores estiverem incorretos.
    """
    expected = [ID_COL, TARGET]
    if list(submission.columns) != expected:
        raise ValueError(f"Colunas esperadas {expected}, obtidas {list(submission.columns)}")
    if len(submission) != len(test):
        raise ValueError(f"Esperadas {len(test)} linhas, obtidas {len(submission)}")
    if set(submission[ID_COL]) != set(test[ID_COL]):
        raise ValueError("PassengerId da submissão não corresponde ao conjunto de teste.")
    if not submission[TARGET].isin([0, 1]).all():
        raise ValueError("Survived deve conter apenas 0 e 1.")


def main(argv: list[str] | None = None) -> None:
    """Treina os modelos finais, salva artefatos e gera as submissões.

    Args:
        argv: Argumentos de linha de comando (None usa sys.argv).
    """
    parser = argparse.ArgumentParser(description="Treina modelos finais e gera submissões.")
    parser.add_argument("models", nargs="*", default=list(DEFAULT_MODELS))
    args = parser.parse_args(argv)

    train, test = build_train_test(*load_raw())
    X, y = train.drop(columns=TARGET), train[TARGET]
    best = load_best_params()
    gender = pd.read_csv(RAW_DIR / "gender_submission.csv")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    submissions: dict[str, pd.DataFrame] = {}
    for name in args.models:
        model = get_final_estimator(name, best).fit(X, y)
        submission = make_submission(model, test)
        validate_submission(submission, test)

        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        submission.to_csv(SUBMISSIONS_DIR / f"{name}.csv", index=False)
        submissions[name] = submission

        merged = submission.merge(gender, on=ID_COL, suffixes=("", "_gender"))
        agreement = (merged[TARGET] == merged[f"{TARGET}_gender"]).mean()
        print(
            f"{name:<14} | sobrevivência prevista {submission[TARGET].mean():.3f} "
            f"(treino {y.mean():.3f}) | concordância c/ gender baseline {agreement:.3f}"
        )

    for a, b in combinations(submissions, 2):
        diff = int((submissions[a][TARGET] != submissions[b][TARGET]).sum())
        print(f"{a} vs {b}: {diff} passageiros com previsões diferentes")


if __name__ == "__main__":
    main()
