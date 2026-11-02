import pandas as pd
import pytest

from titanic.config import TARGET
from titanic.data import FEATURE_COLUMNS, load_csv, load_raw


@pytest.fixture
def raw_dir(tmp_path):
    row = {col: [1] for col in FEATURE_COLUMNS}
    pd.DataFrame(row).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({**row, TARGET: [0]}).to_csv(tmp_path / "train.csv", index=False)
    return tmp_path


def test_load_raw_returns_train_and_test(raw_dir):
    train, test = load_raw(raw_dir)
    assert TARGET in train.columns
    assert TARGET not in test.columns


def test_load_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="make data"):
        load_csv(tmp_path / "nope.csv", FEATURE_COLUMNS)


def test_load_csv_missing_column(raw_dir):
    with pytest.raises(ValueError, match=TARGET):
        load_csv(raw_dir / "test.csv", FEATURE_COLUMNS | {TARGET})
