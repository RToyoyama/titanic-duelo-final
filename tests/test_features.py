import pandas as pd
import pytest

from titanic.config import TARGET
from titanic.features import build_features, build_train_test, extract_title, family_band


def test_extract_title_groups_aliases_and_rare():
    names = pd.Series(["A, Mr. X", "B, Mlle. Y", "C, Mme. Z", "D, Dr. W", "E, Master. V"])
    assert extract_title(names).tolist() == ["Mr", "Miss", "Mrs", "Rare", "Master"]


def test_family_band_boundaries():
    result = family_band(pd.Series([1, 2, 4, 5, 11])).tolist()
    assert result == ["Alone", "Small", "Small", "Large", "Large"]


def test_build_features_values(raw_sample):
    out = build_features(raw_sample)
    assert out["Title"].tolist() == ["Mr", "Mrs", "Master", "Rare"]
    assert out["FamilyBand"].tolist() == ["Small", "Small", "Large", "Alone"]
    assert out["TicketGroupSize"].tolist() == [2, 1, 2, 1]
    assert out["Deck"].tolist() == ["U", "ABC", "U", "ABC"]
    assert out["FarePerPerson"].iloc[0] == pytest.approx(3.625)


def test_build_features_does_not_mutate_input(raw_sample):
    before = raw_sample.copy()
    build_features(raw_sample)
    pd.testing.assert_frame_equal(raw_sample, before)


def test_build_train_test_counts_groups_across_sets(raw_sample):
    train = raw_sample.iloc[[0, 1, 3]]
    test = raw_sample.iloc[[2]].drop(columns=TARGET)
    tr, te = build_train_test(train, test)

    assert (len(tr), len(te)) == (3, 1)
    assert TARGET not in te.columns
    assert tr[TARGET].dtype == int
    assert te["TicketGroupSize"].iloc[0] == 2


def test_build_features_family_columns(raw_sample):
    out = build_features(raw_sample)
    assert out["IsWomanChild"].tolist() == [0, 1, 1, 1]
    assert out["Surname"].iloc[0] == "Braund"


def test_family_key_groups_sequential_tickets(raw_sample):
    family = raw_sample.iloc[:2].assign(
        Name=["Smith, Mrs. Anna", "Smith, Master. Bob"], Ticket=["345773", "345774"]
    )
    out = build_features(family)
    assert out["FamilyKey"].nunique() == 1
