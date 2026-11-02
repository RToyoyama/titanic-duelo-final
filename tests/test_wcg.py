import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from titanic.wcg import WomanChildGroupOverride


@pytest.fixture
def train_data():
    X = pd.DataFrame(
        {
            "FamilyKey": ["F1", "F1", "F2", "F2"],
            "IsWomanChild": [1, 1, 1, 0],
            "Title": ["Mrs", "Miss", "Mrs", "Mr"],
            "Sex": ["female", "female", "female", "male"],
        }
    )
    return X, [1, 1, 0, 1]


@pytest.fixture
def test_data():
    return pd.DataFrame(
        {
            "FamilyKey": ["F1", "F2", "F3"],
            "IsWomanChild": [1, 1, 1],
            "Title": ["Master", "Miss", "Master"],
            "Sex": ["male", "female", "male"],
        }
    )


def test_boy_in_surviving_family_is_saved(train_data, test_data):
    base = DummyClassifier(strategy="constant", constant=0)
    preds = WomanChildGroupOverride(base).fit(*train_data).predict(test_data)
    assert preds.tolist() == [1, 0, 0]


def test_female_in_dead_family_is_lost(train_data, test_data):
    base = DummyClassifier(strategy="constant", constant=1)
    preds = WomanChildGroupOverride(base).fit(*train_data).predict(test_data)
    assert preds.tolist() == [1, 0, 1]


def test_adult_male_label_is_ignored_in_group_rate(train_data):
    model = WomanChildGroupOverride(DummyClassifier()).fit(*train_data)
    assert model.group_rate_ == {"F1": 1.0, "F2": 0.0}
