import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import RepeatedStratifiedKFold

from titanic.config import TARGET
from titanic.features import build_features
from titanic.models import build_pipeline, get_model_specs
from titanic.tuning import (
    SEARCH_SPECS,
    load_best_params,
    run_search,
    save_best_params,
    to_jsonable,
)


def test_every_catalog_model_has_search_spec():
    assert set(SEARCH_SPECS) == set(get_model_specs())


@pytest.mark.parametrize("name", list(SEARCH_SPECS))
def test_search_space_keys_are_valid_params(name):
    valid = build_pipeline(name).get_params()
    assert set(SEARCH_SPECS[name].space) <= set(valid)


def test_to_jsonable_converts_numpy():
    out = to_jsonable({"a": np.int64(3), "b": np.float64(0.5), "c": "scale", "d": None})
    assert out == {"a": 3, "b": 0.5, "c": "scale", "d": None}
    assert type(out["a"]) is int
    json.dumps(out)


def test_save_and_load_roundtrip(tmp_path):
    best = {"svm": {"model__C": 1.0, "model__gamma": "scale"}}
    save_best_params(best, tmp_path / "params.json")
    assert load_best_params(tmp_path / "params.json") == best


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="make tune"):
        load_best_params(tmp_path / "nope.json")


def test_run_search_on_small_data(raw_sample):
    data = build_features(pd.concat([raw_sample] * 10, ignore_index=True))
    X, y = data.drop(columns=TARGET), data[TARGET]
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=0)

    params, score = run_search("logreg", X, y, cv=cv)

    assert set(params) == {"model__C"}
    assert 0 <= score <= 1
