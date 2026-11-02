import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

from titanic.config import TARGET
from titanic.evaluate import evaluate_models, evaluate_pipelines, paired_comparison
from titanic.features import build_features
from titanic.models import build_pipeline


def test_evaluate_models_returns_summary_and_folds(raw_sample):
    data = build_features(pd.concat([raw_sample] * 10, ignore_index=True))
    X, y = data.drop(columns=TARGET), data[TARGET]
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=0)

    summary, folds = evaluate_models(X, y, model_names=["logreg"], cv=cv)

    assert list(summary.index) == ["logreg"]
    assert {"acc_mean", "acc_std", "overfit_gap"} <= set(summary.columns)
    assert len(folds) == 2


def test_paired_comparison_win_rate():
    folds = pd.DataFrame(
        {
            "model": ["logreg"] * 3 + ["svm"] * 3,
            "fold": [0, 1, 2] * 2,
            "accuracy": [0.80, 0.80, 0.80, 0.82, 0.79, 0.83],
        }
    )
    result = paired_comparison(folds)
    assert result.loc["svm", "win_rate"] == round(2 / 3, 4)
    assert result.loc["svm", "mean_diff"] == round(0.0133333, 4)


def test_models_without_scores_get_nan_roc_auc(raw_sample):
    data = build_features(pd.concat([raw_sample] * 10, ignore_index=True))
    X, y = data.drop(columns=TARGET), data[TARGET]
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=0)
    hard = VotingClassifier(
        [("a", build_pipeline("logreg")), ("b", build_pipeline("svm"))], voting="hard"
    )

    summary, _ = evaluate_pipelines({"hard": hard}, X, y, cv=cv)

    assert pd.isna(summary.loc["hard", "roc_auc"])
    assert summary.loc["hard", "acc_mean"] > 0
