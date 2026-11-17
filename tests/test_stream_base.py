"""Tests for numcompute_stream.base (shared base classes)."""

import numpy as np
import pytest

from numcompute_stream.base import StreamingClassifier, StreamingTransformer


class RunningSum(StreamingTransformer):
    """Tiny transformer used only for testing: subtracts the running sum."""

    def __init__(self, offset=0.0):
        self.offset = offset
        self.reset()

    def reset(self):
        self.total_ = None
        return self

    def partial_fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        part = X.sum(axis=0)
        self.total_ = part if self.total_ is None else self.total_ + part
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) - self.total_ + self.offset


class AlwaysOne(StreamingClassifier):
    """Tiny classifier used only for testing: always predicts 1."""

    def __init__(self):
        pass

    def reset(self):
        return self

    def partial_fit(self, X, y):
        return self

    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile([0.0, 1.0], (len(X), 1))


def test_base_classes_cannot_be_created_directly():
    with pytest.raises(TypeError):
        StreamingTransformer()
    with pytest.raises(TypeError):
        StreamingClassifier()


def test_get_params_set_params_and_repr():
    t = RunningSum(offset=2.0)
    assert t.get_params() == {"offset": 2.0}
    assert repr(t) == "RunningSum(offset=2.0)"
    t.set_params(offset=5.0)
    assert t.offset == 5.0
    with pytest.raises(ValueError, match="Invalid parameter 'scale'"):
        t.set_params(scale=1.0)


def test_clone_copies_settings_but_not_learned_state():
    t = RunningSum(offset=3.0).partial_fit([[1.0, 2.0]])
    copy = t.clone()
    assert copy.offset == 3.0
    assert copy.total_ is None
    assert copy is not t


def test_fit_starts_from_scratch_but_partial_fit_accumulates():
    X = [[1.0, 2.0]]
    t = RunningSum()
    t.fit(X)
    t.fit(X)
    assert t.total_.tolist() == [1.0, 2.0]   # fit forgot the first call
    t.partial_fit(X)
    assert t.total_.tolist() == [2.0, 4.0]   # partial_fit added to it


def test_partial_fit_transform_matches_separate_calls():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    a = RunningSum().partial_fit_transform(X)
    b = RunningSum().partial_fit(X).transform(X)
    assert np.array_equal(a, b)


def test_classifier_score_returns_python_float_accuracy():
    score = AlwaysOne().score(np.zeros((4, 2)), [1, 1, 0, 1])
    assert isinstance(score, float)
    assert score == 0.75
