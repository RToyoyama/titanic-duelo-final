import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC

from titanic.config import TARGET
from titanic.ensemble import build_base_estimators, build_ensembles, error_matrix
from titanic.features import build_features

FAST_PARAMS = {
    "logreg": {},
    "svm": {},
    "random_forest": {"model__n_estimators": 10},
    "xgboost": {"model__n_estimators": 10},
}


def test_svm_is_calibrated_only_when_requested():
    with_proba = dict(build_base_estimators(FAST_PARAMS, probability=True))
    without = dict(build_base_estimators(FAST_PARAMS))
    assert isinstance(with_proba["svm"].named_steps["model"], CalibratedClassifierCV)
    assert isinstance(without["svm"].named_steps["model"], SVC)


def test_build_ensembles_types():
    ensembles = build_ensembles(FAST_PARAMS)
    assert set(ensembles) == {"voting_hard", "voting_soft", "stacking"}
    assert isinstance(ensembles["stacking"], StackingClassifier)
    assert ensembles["voting_soft"].voting == "soft"


@pytest.mark.parametrize("kind", ["voting_hard", "voting_soft", "stacking"])
def test_ensembles_fit_and_predict(raw_sample, kind):
    data = build_features(pd.concat([raw_sample] * 10, ignore_index=True))
    X, y = data.drop(columns=TARGET), data[TARGET]
    preds = build_ensembles(FAST_PARAMS)[kind].fit(X, y).predict(X)
    assert len(preds) == len(X)


def test_error_matrix():
    oof = pd.DataFrame({"a": [1, 0, 1, 0], "b": [1, 1, 1, 0]})
    errors = error_matrix(oof, pd.Series([1, 0, 0, 0]))
    assert errors["a"].tolist() == [0, 0, 1, 0]
    assert errors["b"].tolist() == [0, 1, 1, 0]
