import numpy as np
import pytest
from sklearn.base import clone

from titanic.config import TARGET
from titanic.features import build_features
from titanic.preprocessing import build_preprocessor


@pytest.fixture
def featured(raw_sample):
    return build_features(raw_sample)


def test_output_has_no_nan(featured):
    out = build_preprocessor().fit_transform(featured, featured[TARGET])
    assert len(out) == len(featured)
    assert not out.isna().any().any()


def test_expected_feature_names(featured):
    out = build_preprocessor().fit_transform(featured, featured[TARGET])
    assert {"Age", "FarePerPerson", "group_survival", "Sex_female"} <= set(out.columns)
    assert "Name" not in out.columns


def test_group_survival_can_be_disabled(featured):
    out = build_preprocessor(group_survival=False).fit_transform(featured, featured[TARGET])
    assert "group_survival" not in out.columns


def test_no_scale_keeps_log_fare(featured):
    out = build_preprocessor(scale=False).fit_transform(featured, featured[TARGET])
    np.testing.assert_allclose(out["FarePerPerson"], np.log1p(featured["FarePerPerson"]))


def test_transform_unseen_data_without_target(featured):
    prep = build_preprocessor().fit(featured.iloc[:3], featured[TARGET].iloc[:3])
    out = prep.transform(featured.iloc[3:].drop(columns=TARGET))
    assert out.shape[0] == 1
    assert not out.isna().any().any()


def test_preprocessor_is_cloneable():
    clone(build_preprocessor())
