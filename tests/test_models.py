import pandas as pd
import pytest

from titanic.config import TARGET
from titanic.features import build_features
from titanic.models import build_pipeline, get_model_specs


@pytest.mark.parametrize("name", list(get_model_specs()))
def test_every_model_fits_and_predicts(featured, name):
    X, y = featured.drop(columns=TARGET), featured[TARGET]
    preds = build_pipeline(name).fit(X, y).predict(X)
    assert len(preds) == len(X)


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="Opções"):
        build_pipeline("modelo_inexistente")


def test_tree_models_skip_scaling():
    prep = build_pipeline("random_forest").named_steps["prep"]
    num_pipeline = prep.transformers[0][1]
    assert num_pipeline.named_steps["scale"] == "passthrough"


def test_specs_return_fresh_instances():
    assert get_model_specs()["svm"].estimator is not get_model_specs()["svm"].estimator


def test_params_are_applied():
    pipe = build_pipeline("svm", params={"model__C": 3.0})
    assert pipe.get_params()["model__C"] == 3.0


def test_smote_adds_sampler_step():
    assert "smote" in build_pipeline("logreg", balance="smote").named_steps


@pytest.mark.parametrize("name", list(get_model_specs()))
def test_class_weight_is_applied(name):
    params = build_pipeline(name, balance="class_weight").get_params()
    key = "model__scale_pos_weight" if name == "xgboost" else "model__class_weight"
    assert params[key] not in (None, 1)


def test_invalid_balance_raises():
    with pytest.raises(ValueError, match="Balanceamento"):
        build_pipeline("logreg", balance="oversample")


def test_smote_pipeline_fits(raw_sample):
    data = build_features(pd.concat([raw_sample] * 10, ignore_index=True))
    X, y = data.drop(columns=TARGET), data[TARGET]
    preds = build_pipeline("logreg", balance="smote").fit(X, y).predict(X)
    assert len(preds) == len(X)
