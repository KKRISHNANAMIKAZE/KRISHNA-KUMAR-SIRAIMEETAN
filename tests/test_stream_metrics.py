"""Tests for numcompute_stream.metrics (streaming metrics)."""

import warnings

import numpy as np
import pytest

from numcompute_stream.io import chunk_array
from numcompute_stream.metrics import (
    ROCAUC,
    F1,
    Accuracy,
    ConfusionMatrix,
    Precision,
    Recall,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def random_labels(n=400, k=3, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, k, n)
    y_pred = np.where(rng.random(n) < 0.7, y_true, rng.integers(0, k, n))
    return y_true, y_pred


def stream(metric, a, b, chunk_size=33):
    for a_chunk, b_chunk in chunk_array(a, b, chunk_size=chunk_size):
        metric.update(a_chunk, b_chunk)
    return metric


def pairwise_auc(y, s):
    """Slow reference AUC: compare every positive with every negative."""
    pos, neg = s[y == 1], s[y != 1]
    diff = pos[:, None] - neg[None, :]
    return (np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size


def test_confusion_matrix_known_values():
    cm = confusion_matrix([0, 0, 1, 1, 2], [0, 1, 1, 1, 0])
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]


def test_streamed_confusion_matrix_equals_batch():
    y_true, y_pred = random_labels()
    streamed = stream(ConfusionMatrix(), y_true, y_pred).result()
    assert np.array_equal(streamed, confusion_matrix(y_true, y_pred))
    assert streamed.sum() == len(y_true)


def test_confusion_matrix_grows_when_a_new_label_appears():
    cm = ConfusionMatrix()
    cm.update(["cat", "dog"], ["cat", "cat"])
    cm.update(["fox"], ["dog"])                       # new label "fox"
    assert cm.labels_.tolist() == ["cat", "dog", "fox"]
    assert cm.result().tolist() == [[1, 0, 0], [1, 0, 0], [0, 1, 0]]


def test_confusion_matrix_fixed_labels_reject_unknown():
    cm = ConfusionMatrix(labels=[0, 1])
    with pytest.raises(ValueError, match=r"\[2\] are not in the fixed labels"):
        cm.update([0, 2], [0, 1])


@pytest.mark.parametrize("average", ["macro", "micro", "weighted"])
def test_streamed_precision_recall_f1_equal_batch(average):
    y_true, y_pred = random_labels()
    for cls, func in ((Precision, precision_score), (Recall, recall_score),
                      (F1, f1_score)):
        streamed = stream(cls(average=average), y_true, y_pred).result()
        assert streamed == pytest.approx(func(y_true, y_pred, average=average))


def test_hand_computed_binary_scores():
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 1, 0, 1, 0]          # TP=2, FN=1, FP=1
    assert precision_score(y_true, y_pred, average="binary") == pytest.approx(2 / 3)
    assert recall_score(y_true, y_pred, average="binary") == pytest.approx(2 / 3)
    assert f1_score(y_true, y_pred, average="binary") == pytest.approx(2 / 3)
    assert accuracy_score(y_true, y_pred) == pytest.approx(0.6)


def test_micro_average_equals_accuracy():
    y_true, y_pred = random_labels()
    assert precision_score(y_true, y_pred, average="micro") == pytest.approx(
        accuracy_score(y_true, y_pred))


def test_class_never_predicted_scores_zero_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert precision_score([0, 1], [0, 0], average="binary") == 0.0
        assert f1_score([0, 1], [0, 0], average="binary") == 0.0


def test_rolling_window_only_counts_recent_chunks():
    acc = Accuracy(window=2)
    acc.update([1, 1], [0, 0])        # 0% - will drop out of the window
    acc.update([1, 1], [1, 1])
    acc.update([1, 1], [1, 0])
    assert acc.result() == pytest.approx(0.75)
    whole = Accuracy()
    for t, p in (([1, 1], [0, 0]), ([1, 1], [1, 1]), ([1, 1], [1, 0])):
        whole.update(t, p)
    assert whole.result() == pytest.approx(0.5)


def test_metric_reset_and_empty_result():
    acc = Accuracy().update([1], [1])
    assert acc.result() == 1.0
    assert np.isnan(acc.reset().result())
    assert np.isnan(F1().result())


def test_metric_input_errors():
    with pytest.raises(ValueError, match="y_true=2, y_pred=3"):
        Accuracy().update([1, 0], [1, 0, 1])
    with pytest.raises(ValueError, match="NaN"):
        Accuracy().update([1.0, np.nan], [1.0, 1.0])
    with pytest.raises(ValueError, match="average"):
        Precision(average="sideways")


def test_exact_auc_matches_pairwise_reference_with_ties():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.random(300) * 0.5 + y * 0.3, 2)          # many ties
    assert roc_auc_score(y, s) == pytest.approx(pairwise_auc(y, s))


def test_streamed_binned_auc_is_close_to_exact():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 2000)
    s = np.clip(rng.normal(0.4 + 0.2 * y, 0.15), 0, 1)
    streamed = stream(ROCAUC(), y, s, chunk_size=150).result()
    assert streamed == pytest.approx(roc_auc_score(y, s), abs=1e-3)
    coarse = np.round(s, 1)               # scores exactly on bin edges
    assert stream(ROCAUC(), y, coarse).result() == pytest.approx(
        roc_auc_score(y, coarse), abs=1e-12)


def test_auc_needs_both_classes_and_valid_scores():
    auc = ROCAUC().update([1, 1], [0.2, 0.9])
    assert np.isnan(auc.result())                            # no negatives yet
    auc.update([0], [0.1])
    assert auc.result() == 1.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ROCAUC().update([1, 0], [1.2, 0.3])
    assert ROCAUC.needs_proba and not Accuracy.needs_proba
