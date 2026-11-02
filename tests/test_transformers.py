import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from titanic.transformers import GroupMedianImputer, GroupSurvivalEncoder


@pytest.fixture
def tickets():
    return pd.DataFrame({"Ticket": ["A", "A", "A", "B"]})


def test_fit_transform_uses_leave_one_out(tickets):
    out = GroupSurvivalEncoder(default=-1.0).fit_transform(tickets, [1, 1, 0, 1])
    np.testing.assert_allclose(out.ravel(), [0.5, 0.5, 1.0, -1.0])


def test_own_label_does_not_leak(tickets):
    enc = GroupSurvivalEncoder()
    a = enc.fit_transform(tickets, [1, 1, 0, 1])
    b = clone(enc).fit_transform(tickets, [0, 1, 0, 1])
    assert a[0, 0] == b[0, 0]


def test_transform_unseen_rows_use_full_group(tickets):
    enc = GroupSurvivalEncoder(default=-1.0).fit(tickets, [1, 1, 0, 1])
    out = enc.transform(pd.DataFrame({"Ticket": ["A", "C"]}))
    np.testing.assert_allclose(out.ravel(), [2 / 3, -1.0])


def test_fit_transform_requires_y(tickets):
    with pytest.raises(ValueError, match="requer y"):
        GroupSurvivalEncoder().fit_transform(tickets)


def test_group_median_imputer_fills_by_group_then_global():
    X = pd.DataFrame(
        {
            "Title": ["Mr", "Mr", "Master", "Master", "Miss"],
            "Age": [30.0, np.nan, 4.0, np.nan, np.nan],
        }
    )
    out = GroupMedianImputer().fit_transform(X)
    np.testing.assert_allclose(out.ravel(), [30, 30, 4, 4, 17])
