"""Tests for numcompute_stream.utils (shared input validation)."""

import numpy as np
import pytest

from numcompute_stream.utils import (
    NotFittedError,
    check_is_fitted,
    check_labels,
    check_n_features,
    check_positive_int,
    check_random_state,
    check_X,
    check_X_y,
)


def test_check_X_converts_int_and_bool_to_float():
    assert check_X([[1, 2], [3, 4]]).dtype == np.float64
    out = check_X(np.array([[True, False]]))
    assert out.dtype == np.float64
    assert out.tolist() == [[1.0, 0.0]]


def test_check_X_does_not_modify_callers_array():
    X = np.array([[1.0, 2.0]])
    out = check_X(X)
    out[0, 0] = 99.0
    assert X[0, 0] == 1.0


def test_check_X_rejects_1d_input_with_reshape_hint():
    with pytest.raises(ValueError, match=r"reshape\(-1, 1\)"):
        check_X([1.0, 2.0, 3.0])


def test_check_X_rejects_3d_input_and_zero_columns():
    with pytest.raises(ValueError, match="must be 2-D"):
        check_X(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="at least one feature"):
        check_X(np.zeros((3, 0)))


def test_check_X_rejects_non_numeric_data():
    with pytest.raises(TypeError, match="numeric"):
        check_X([["a", "b"]])


def test_check_X_empty_chunk_allowed_only_when_requested():
    assert check_X(np.zeros((0, 3))).shape == (0, 3)
    with pytest.raises(ValueError, match="at least one sample"):
        check_X(np.zeros((0, 3)), allow_empty=False)


def test_check_X_nan_policy():
    X = [[1.0, np.nan]]
    assert np.isnan(check_X(X)[0, 1])
    with pytest.raises(ValueError, match="Imputer"):
        check_X(X, allow_nan=False)


def test_check_X_rejects_infinity():
    with pytest.raises(ValueError, match="infinite"):
        check_X([[1.0, np.inf]])


def test_check_labels_flattens_column_vector_and_keeps_strings():
    assert check_labels([[0], [1], [1]]).shape == (3,)
    assert check_labels(["cat", "dog"]).tolist() == ["cat", "dog"]


def test_check_labels_rejects_nan_none_and_matrix():
    with pytest.raises(ValueError, match="NaN"):
        check_labels([0.0, np.nan])
    with pytest.raises(ValueError, match="None"):
        check_labels(np.array(["a", None], dtype=object))
    with pytest.raises(ValueError, match="1-D"):
        check_labels(np.zeros((3, 2)))


def test_check_X_y_reports_both_lengths():
    with pytest.raises(ValueError, match="X=3, y=2"):
        check_X_y(np.zeros((3, 2)), [0, 1])


def test_check_n_features_message_names_both_counts():
    with pytest.raises(ValueError, match="X has 2 features, but Scaler was "
                                         "fitted with 3"):
        check_n_features(np.zeros((1, 2)), 3, owner="Scaler")


def test_check_is_fitted_raises_not_fitted_error():
    class Model:
        mean_ = None

    with pytest.raises(NotFittedError, match="Model instance is not fitted"):
        check_is_fitted(Model(), "mean_")
    # Callers catching the standard errors still catch it.
    assert issubclass(NotFittedError, ValueError)
    assert issubclass(NotFittedError, AttributeError)


def test_check_positive_int_rules():
    assert check_positive_int(5, "buffer_size") == 5
    assert check_positive_int(None, "max_depth", allow_none=True) is None
    with pytest.raises(TypeError):
        check_positive_int(True, "buffer_size")
    with pytest.raises(TypeError):
        check_positive_int(2.5, "buffer_size")
    with pytest.raises(ValueError, match="at least 2"):
        check_positive_int(1, "min_samples_split", minimum=2)


def test_check_random_state_is_repeatable():
    a = check_random_state(7).random(3)
    b = check_random_state(7).random(3)
    assert np.array_equal(a, b)
    rng = np.random.default_rng(0)
    assert check_random_state(rng) is rng
    with pytest.raises(TypeError):
        check_random_state("seed")
