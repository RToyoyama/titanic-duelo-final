import pandas as pd
import pytest


@pytest.fixture
def raw_sample():
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Survived": [0, 1, 1, 0],
            "Pclass": [3, 1, 3, 2],
            "Name": [
                "Braund, Mr. Owen",
                "Cumings, Mrs. John",
                "Palsson, Master. Gosta",
                "Rothes, the Countess. of",
            ],
            "Sex": ["male", "female", "male", "female"],
            "Age": [22.0, 38.0, None, 33.0],
            "SibSp": [1, 1, 3, 0],
            "Parch": [0, 0, 1, 0],
            "Ticket": ["A/5", "PC1", "A/5", "X9"],
            "Fare": [7.25, 71.28, 21.07, 86.5],
            "Cabin": [None, "C85", None, "B77"],
            "Embarked": ["S", "C", "S", "S"],
        }
    )
