import pandas as pd
import pytest
from sklearn.ensemble import VotingClassifier

from titanic.config import ID_COL, TARGET
from titanic.features import build_features
from titanic.predict import get_final_estimator, make_submission, validate_submission

FAST_PARAMS = {
    "logreg": {},
    "svm": {},
    "random_forest": {"model__n_estimators": 10},
    "xgboost": {"model__n_estimators": 10},
}


@pytest.fixture
def fitted_and_test(raw_sample):
    data = build_features(pd.concat([raw_sample] * 10, ignore_index=True))
    model = get_final_estimator("logreg", FAST_PARAMS)
    model.fit(data.drop(columns=TARGET), data[TARGET])
    test = build_features(raw_sample).drop(columns=TARGET)
    return model, test


def test_get_final_estimator_resolves_catalog_and_ensembles():
    assert "model" in get_final_estimator("logreg", FAST_PARAMS).named_steps
    assert isinstance(get_final_estimator("voting_hard", FAST_PARAMS), VotingClassifier)


def test_get_final_estimator_unknown_raises():
    with pytest.raises(KeyError, match="desconhecido"):
        get_final_estimator("modelo_inexistente", FAST_PARAMS)


def test_valid_submission_passes(fitted_and_test):
    model, test = fitted_and_test
    submission = make_submission(model, test)
    validate_submission(submission, test)
    assert submission[TARGET].dtype == int


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (lambda s: s.rename(columns={TARGET: "survived"}), "Colunas"),
        (lambda s: s.iloc[:-1], "linhas"),
        (lambda s: s.assign(**{TARGET: 2}), "0 e 1"),
        (lambda s: s.assign(**{ID_COL: s[ID_COL] + 1000}), "PassengerId"),
    ],
)
def test_invalid_submission_raises(fitted_and_test, corrupt, message):
    model, test = fitted_and_test
    with pytest.raises(ValueError, match=message):
        validate_submission(corrupt(make_submission(model, test)), test)
