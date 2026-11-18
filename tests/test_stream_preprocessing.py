"""Tests for numcompute_stream.preprocessing (streaming transformers)."""

import numpy as np
import pytest

from numcompute_stream.io import chunk_array
from numcompute_stream.preprocessing import Imputer, OneHotEncoder, StandardScaler
from numcompute_stream.utils import NotFittedError


def stream_fit(transformer, X, chunk_size=25):
    for chunk, _ in chunk_array(X, chunk_size=chunk_size):
        transformer.partial_fit(chunk)
    return transformer


# ---------------------------------------------------------------- scaler

def test_scaler_streaming_matches_batch_statistics():
    X = np.random.default_rng(0).normal(3.0, 4.0, size=(203, 3))
    scaler = stream_fit(StandardScaler(), X)
    assert np.allclose(scaler.mean_, X.mean(axis=0))
    assert np.allclose(scaler.scale_, X.std(axis=0))
    Z = scaler.transform(X)
    assert np.allclose(Z.mean(axis=0), 0.0)
    assert np.allclose(Z.std(axis=0), 1.0)


def test_scaler_constant_column_gives_zeros_not_nan():
    # Regression test for the Assignment 1 bug (division by zero).
    X = np.array([[1.0, 5.0], [3.0, 5.0]])
    Z = StandardScaler().fit_transform(X)
    assert not np.isnan(Z).any()
    assert Z[:, 1].tolist() == [0.0, 0.0]


def test_scaler_ignores_nan_when_learning_and_keeps_it():
    X = np.array([[1.0], [np.nan], [3.0]])
    scaler = StandardScaler().fit(X)
    assert scaler.mean_[0] == 2.0
    assert np.isnan(scaler.transform(X)[1, 0])


def test_scaler_rejects_wrong_number_of_features():
    # Regression test for the Assignment 1 bug (silent broadcasting).
    scaler = StandardScaler().fit(np.ones((3, 2)))
    with pytest.raises(ValueError, match="StandardScaler was fitted with 2"):
        scaler.transform(np.ones((3, 1)))
    with pytest.raises(ValueError, match="StandardScaler was fitted with 2"):
        scaler.partial_fit(np.ones((3, 3)))


def test_transform_before_fit_gives_clear_error():
    with pytest.raises(NotFittedError, match="StandardScaler instance is not"):
        StandardScaler().transform([[1.0]])
    with pytest.raises(NotFittedError):
        Imputer().transform([[1.0]])


def test_scaler_inverse_transform_round_trip_and_fit_resets():
    X = np.random.default_rng(1).normal(size=(50, 2))
    scaler = StandardScaler().fit(X * 100)
    scaler.fit(X)                                      # fit starts again
    assert np.allclose(scaler.mean_, X.mean(axis=0))
    assert np.allclose(scaler.inverse_transform(scaler.transform(X)), X)


# ---------------------------------------------------------------- imputer

def test_imputer_mean_updates_on_the_fly():
    imp = Imputer(strategy="mean")
    imp.partial_fit(np.array([[1.0], [np.nan]]))
    assert imp.transform([[np.nan]])[0, 0] == 1.0
    imp.partial_fit(np.array([[5.0]]))
    assert imp.transform([[np.nan]])[0, 0] == 3.0     # mean of 1 and 5


def test_imputer_median_constant_and_never_seen_column():
    X = np.array([[1.0, np.nan], [2.0, np.nan], [10.0, np.nan]])
    median = Imputer(strategy="median", fill_value=-1.0).fit(X)
    assert median.statistics_.tolist() == [2.0, -1.0]  # column 2: nothing seen
    const = Imputer(strategy="constant", fill_value=7.0).fit(X)
    assert not np.isnan(const.transform(X)).any()
    assert const.transform(X)[0, 1] == 7.0
    with pytest.raises(ValueError, match="strategy"):
        Imputer(strategy="mode")


# ---------------------------------------------------------------- one-hot

def test_onehot_encodes_and_names_columns():
    X = np.array([["red", "S"], ["blue", "M"], ["red", "M"]])
    enc = OneHotEncoder().fit(X)
    assert enc.get_feature_names_out(["colour", "size"]) == [
        "colour=blue", "colour=red", "size=M", "size=S"]
    assert enc.transform(X).tolist() == [[0, 1, 0, 1], [1, 0, 1, 0],
                                         [0, 1, 1, 0]]


def test_onehot_new_category_is_added_at_the_end():
    enc = OneHotEncoder().partial_fit(np.array([["red"], ["blue"]]))
    before = enc.transform(np.array([["red"], ["blue"]]))
    enc.partial_fit(np.array([["green"]]))
    after = enc.transform(np.array([["red"], ["blue"], ["green"]]))
    assert enc.n_features_out_ == 3
    assert np.array_equal(after[:2, :2], before)       # old columns unchanged
    assert after[2].tolist() == [0.0, 0.0, 1.0]


def test_onehot_unknown_category_error_or_zeros():
    X = np.array([["a"], ["b"]])
    with pytest.raises(ValueError, match=r"not seen during fitting: \['c'\]"):
        OneHotEncoder().fit(X).transform(np.array([["c"]]))
    ignore = OneHotEncoder(handle_unknown="ignore").fit(X)
    assert ignore.transform(np.array([["c"]])).tolist() == [[0.0, 0.0]]


def test_onehot_fixed_categories_and_input_errors():
    enc = OneHotEncoder(categories=[["low", "high"]])
    enc.partial_fit(np.array([["high"]]))
    assert enc.transform(np.array([["low"]])).tolist() == [[1.0, 0.0]]
    with pytest.raises(ValueError, match="unknown categories"):
        enc.partial_fit(np.array([["medium"]]))
    with pytest.raises(ValueError, match="reshape"):
        OneHotEncoder().fit(np.array(["a", "b"]))
    with pytest.raises(ValueError, match="NaN"):
        OneHotEncoder().fit(np.array([[1.0], [np.nan]]))
