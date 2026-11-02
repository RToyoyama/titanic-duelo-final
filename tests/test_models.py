import pytest

from titanic.config import TARGET
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
