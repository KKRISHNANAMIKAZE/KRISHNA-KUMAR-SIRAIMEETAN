"""
Streaming classification metrics.

Every metric class follows :class:`~numcompute_stream.base.StreamingMetric`:

* ``update(y_true_chunk, y_pred_chunk)`` adds one chunk,
* ``result()`` returns the value over everything seen since the last reset,
* ``reset()`` starts again.

Each metric also takes ``window``: with ``window=None`` (default) it covers
the whole stream; with ``window=k`` it covers only the last ``k`` chunks (a
rolling metric), which shows how the model is doing *now*.

How they work
-------------
:class:`ConfusionMatrix` keeps counts of (true class, predicted class)
pairs, built per chunk with one ``np.bincount`` call. Accuracy, precision,
recall and F1 are all computed from that matrix, so they share one tested
counting routine. New class labels can appear at any time; the matrix grows
and old counts are moved to the right rows and columns.

:class:`ROCAUC` keeps two histograms of predicted scores (positives and
negatives) over fixed bins on [0, 1]. The ROC curve and its area come from
running totals of those histograms, so memory stays fixed however long the
stream is. Scores in the same bin count as ties (half credit), which is
exactly how the rank-based AUC treats equal scores. The exact batch AUC is
available as :func:`roc_auc_score` for checking.

Zero-division rule: if a class is never predicted (or never occurs), its
precision (or recall) is 0.0, with no warning. A metric with no data yet
returns NaN.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from numcompute.rank import rankdata  # Assignment 1 ranking, reused for AUC

from .base import StreamingMetric
from .utils import (
    check_consistent_length,
    check_labels,
    check_positive_int,
    safe_divide,
)

__all__ = [
    "Accuracy",
    "ConfusionMatrix",
    "F1",
    "Precision",
    "ROCAUC",
    "Recall",
    "accuracy_score",
    "confusion_matrix",
    "f1_score",
    "precision_score",
    "recall_score",
    "roc_auc_score",
]

_AVERAGES = ("binary", "macro", "micro", "weighted")


def _label_kind(labels):
    return "text" if labels.dtype.kind in "USO" else "number"


def _check_pair(y_true, y_pred):
    y_true = check_labels(y_true, name="y_true")
    y_pred = check_labels(y_pred, name="y_pred")
    check_consistent_length(y_true, y_pred, names=("y_true", "y_pred"))
    return y_true, y_pred


# ----------------------------------------------------------------------
# Streaming metrics
# ----------------------------------------------------------------------

class ConfusionMatrix(StreamingMetric):
    """Confusion matrix accumulated over a stream.

    Parameters
    ----------
    labels : array-like or None, default=None
        Fixed class labels (rows and columns, in this order). If None, the
        labels are learned from the data and the matrix grows when a new
        label appears.
    window : int or None, default=None
        Count only the last ``window`` chunks.

    Attributes
    ----------
    labels_ : ndarray
        Row and column labels of the matrix.

    Notes
    -----
    Rows are true classes and columns are predicted classes:
    ``result()[i, j]`` counts samples of class ``labels_[i]`` predicted as
    ``labels_[j]``.
    """

    def __init__(self, labels=None, window=None):
        self.labels = labels
        self.window = window
        check_positive_int(window, "window", allow_none=True)
        self.reset()

    def reset(self):
        """Clear all counts. Returns ``self``."""
        if self.labels is None:
            self.labels_ = None
        else:
            self.labels_ = check_labels(self.labels, name="labels")
            if len(np.unique(self.labels_)) != len(self.labels_):
                raise ValueError("labels must not contain duplicates")
        self._chunks = deque(maxlen=self.window)
        self.n_seen_ = 0
        return self

    def _grow_labels(self, values):
        new = np.unique(values)
        if self.labels_ is None:
            self.labels_ = new
            return
        if _label_kind(new) != _label_kind(self.labels_):
            raise TypeError(
                "Labels changed type between chunks (text vs numbers)"
            )
        unknown = new[~np.isin(new, self.labels_)]
        if unknown.size == 0:
            return
        if self.labels is not None:
            raise ValueError(
                f"Labels {unknown.tolist()} are not in the fixed labels "
                f"{self.labels_.tolist()}"
            )
        merged = np.union1d(self.labels_, unknown)
        pos = np.searchsorted(merged, self.labels_)
        k = len(merged)
        resized = []
        for m in self._chunks:
            bigger = np.zeros((k, k), dtype=np.int64)
            bigger[np.ix_(pos, pos)] = m
            resized.append(bigger)
        self._chunks = deque(resized, maxlen=self.window)
        self.labels_ = merged

    def update(self, y_true, y_pred):
        """Add one chunk of true and predicted labels.

        Parameters
        ----------
        y_true, y_pred : array-like of shape (n_chunk,)

        Returns
        -------
        self
        """
        y_true, y_pred = _check_pair(y_true, y_pred)
        if y_true.size == 0:
            return self
        self._grow_labels(np.concatenate([y_true, y_pred]))
        k = len(self.labels_)
        t = np.searchsorted(self.labels_, y_true)
        p = np.searchsorted(self.labels_, y_pred)
        counts = np.bincount(t * k + p, minlength=k * k).reshape(k, k)
        if self.window is None and self._chunks:
            self._chunks[0] = self._chunks[0] + counts   # keep one running total
        else:
            self._chunks.append(counts)
        self.n_seen_ += y_true.size
        return self

    def result(self):
        """Return the matrix as an int array of shape (n_labels, n_labels)."""
        if not self._chunks:
            k = 0 if self.labels_ is None else len(self.labels_)
            return np.zeros((k, k), dtype=np.int64)
        return np.sum(self._chunks, axis=0)


class _ConfusionMetric(StreamingMetric):
    """Shared code for metrics computed from a confusion matrix."""

    def __init__(self, window=None):
        self.window = window
        self.reset()

    def reset(self):
        self._cm = ConfusionMatrix(window=self.window)
        return self

    def update(self, y_true, y_pred):
        """Add one chunk of true and predicted labels. Returns ``self``."""
        self._cm.update(y_true, y_pred)
        return self

    @property
    def labels_(self):
        return self._cm.labels_


class Accuracy(_ConfusionMetric):
    """Fraction of correct predictions over the stream (or window).

    Parameters
    ----------
    window : int or None, default=None
        Count only the last ``window`` chunks.
    """

    def result(self):
        """Accuracy as a float in [0, 1], or NaN before any data."""
        cm = self._cm.result()
        total = cm.sum()
        return float("nan") if total == 0 else float(np.trace(cm) / total)


class _PerClassMetric(_ConfusionMetric):
    """Precision, recall and F1 with the usual averaging options.

    Parameters
    ----------
    average : {"macro", "micro", "weighted", "binary"}, default="macro"
        "macro": plain mean over classes. "weighted": mean weighted by how
        often each class truly occurs. "micro": computed from total counts
        (equals accuracy for single-label data). "binary": the score of
        ``pos_label`` only.
    pos_label : label, default=1
        Positive class for ``average="binary"``.
    window : int or None, default=None
        Count only the last ``window`` chunks.
    """

    def __init__(self, average="macro", pos_label=1, window=None):
        if average not in _AVERAGES:
            raise ValueError(f"average must be one of {_AVERAGES}; got {average!r}")
        self.average = average
        self.pos_label = pos_label
        super().__init__(window=window)

    def _per_class(self, tp, predicted, actual):
        raise NotImplementedError

    def result(self):
        """The averaged score as a float, or NaN before any data."""
        cm = self._cm.result()
        if cm.sum() == 0:
            return float("nan")
        tp = np.diag(cm).astype(float)
        predicted = cm.sum(axis=0).astype(float)
        actual = cm.sum(axis=1).astype(float)
        if self.average == "micro":
            return float(self._per_class(tp.sum(), predicted.sum(),
                                         actual.sum()))
        scores = self._per_class(tp, predicted, actual)
        if self.average == "binary":
            hit = np.flatnonzero(self._cm.labels_ == self.pos_label)
            return float(scores[hit[0]]) if hit.size else 0.0
        if self.average == "weighted":
            return float(np.sum(scores * actual) / actual.sum())
        return float(scores.mean())


class Precision(_PerClassMetric):
    """Precision = TP / (TP + FP), streamed. See :class:`_PerClassMetric`."""

    def _per_class(self, tp, predicted, actual):
        return safe_divide(tp, predicted, zero_policy="zero")


class Recall(_PerClassMetric):
    """Recall = TP / (TP + FN), streamed. See :class:`_PerClassMetric`."""

    def _per_class(self, tp, predicted, actual):
        return safe_divide(tp, actual, zero_policy="zero")


class F1(_PerClassMetric):
    """F1 = harmonic mean of precision and recall, streamed."""

    def _per_class(self, tp, predicted, actual):
        p = safe_divide(tp, predicted, zero_policy="zero")
        r = safe_divide(tp, actual, zero_policy="zero")
        return safe_divide(2 * p * r, p + r, zero_policy="zero")


class ROCAUC(StreamingMetric):
    """Area under the ROC curve for binary problems, streamed with bins.

    Parameters
    ----------
    pos_label : label, default=1
        The positive class. Every other label counts as negative.
    n_bins : int, default=1000
        Number of equal-width score bins on [0, 1]. Scores closer together
        than ``1 / n_bins`` may share a bin and then count as a tie, so the
        result can differ from the exact AUC by a tiny amount.
    window : int or None, default=None
        Count only the last ``window`` chunks.

    Notes
    -----
    ``update`` takes the predicted *probability of the positive class*
    (for example a column of ``predict_proba``), not predicted labels.
    """

    needs_proba = True

    def __init__(self, pos_label=1, n_bins=1000, window=None):
        self.pos_label = pos_label
        self.n_bins = n_bins
        self.window = window
        check_positive_int(n_bins, "n_bins", minimum=2)
        check_positive_int(window, "window", allow_none=True)
        self.reset()

    def reset(self):
        """Clear all counts. Returns ``self``."""
        self._chunks = deque(maxlen=self.window)
        return self

    def update(self, y_true, y_score):
        """Add one chunk of true labels and positive-class scores.

        Parameters
        ----------
        y_true : array-like of shape (n_chunk,)
        y_score : array-like of shape (n_chunk,)
            Scores in [0, 1]; higher means "more likely positive".

        Returns
        -------
        self
        """
        y_true = check_labels(y_true, name="y_true")
        y_score = np.asarray(y_score, dtype=float).ravel()
        check_consistent_length(y_true, y_score, names=("y_true", "y_score"))
        if y_true.size == 0:
            return self
        if np.isnan(y_score).any() or y_score.min() < 0 or y_score.max() > 1:
            raise ValueError("y_score must be probabilities in [0, 1]")
        bins = np.minimum((y_score * self.n_bins).astype(np.intp),
                          self.n_bins - 1)
        is_pos = y_true == self.pos_label
        pos = np.bincount(bins[is_pos], minlength=self.n_bins)
        neg = np.bincount(bins[~is_pos], minlength=self.n_bins)
        if self.window is None and self._chunks:
            old_pos, old_neg = self._chunks[0]
            self._chunks[0] = (old_pos + pos, old_neg + neg)
        else:
            self._chunks.append((pos, neg))
        return self

    def result(self):
        """AUC as a float in [0, 1]; NaN until both classes have been seen."""
        if not self._chunks:
            return float("nan")
        pos = np.sum([c[0] for c in self._chunks], axis=0)
        neg = np.sum([c[1] for c in self._chunks], axis=0)
        n_pos, n_neg = pos.sum(), neg.sum()
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        # Walk thresholds from the highest bin down to build the ROC curve.
        tpr = np.concatenate([[0.0], np.cumsum(pos[::-1]) / n_pos])
        fpr = np.concatenate([[0.0], np.cumsum(neg[::-1]) / n_neg])
        return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


# ----------------------------------------------------------------------
# Batch functions (whole arrays at once)
# ----------------------------------------------------------------------

def confusion_matrix(y_true, y_pred, labels=None):
    """Confusion matrix of two label arrays (rows true, columns predicted).

    Returns
    -------
    ndarray of shape (n_labels, n_labels)
        Labels are ``labels`` if given, otherwise the sorted union of both
        inputs.
    """
    return ConfusionMatrix(labels=labels).update(y_true, y_pred).result()


def accuracy_score(y_true, y_pred):
    """Fraction of matching labels (NaN for empty input)."""
    return Accuracy().update(y_true, y_pred).result()


def precision_score(y_true, y_pred, average="macro", pos_label=1):
    """Precision; see :class:`Precision` for ``average``."""
    return Precision(average, pos_label).update(y_true, y_pred).result()


def recall_score(y_true, y_pred, average="macro", pos_label=1):
    """Recall; see :class:`Recall` for ``average``."""
    return Recall(average, pos_label).update(y_true, y_pred).result()


def f1_score(y_true, y_pred, average="macro", pos_label=1):
    """F1 score; see :class:`F1` for ``average``."""
    return F1(average, pos_label).update(y_true, y_pred).result()


def roc_auc_score(y_true, y_score, pos_label=1):
    """Exact binary ROC AUC using ranks (Mann-Whitney U statistic).

    AUC is the chance that a random positive gets a higher score than a
    random negative, with ties counting half. Using average ranks
    (:func:`numcompute.rank.rankdata` from Assignment 1)::

        AUC = (sum of positive ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    Returns
    -------
    float
        NaN if only one class is present.
    """
    y_true = check_labels(y_true, name="y_true")
    y_score = np.asarray(y_score, dtype=float).ravel()
    check_consistent_length(y_true, y_score, names=("y_true", "y_score"))
    is_pos = y_true == pos_label
    n_pos, n_neg = int(is_pos.sum()), int((~is_pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(y_score, method="average")
    return float((ranks[is_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
