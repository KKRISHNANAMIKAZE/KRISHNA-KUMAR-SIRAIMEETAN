"""
Shared input validation for every ``numcompute_stream`` module.

Every estimator, transformer and metric in the package checks its inputs
through the helpers in this file, so the same mistake always produces the
same clear error message. The low-level numeric checks are reused from the
Assignment 1 module :mod:`numcompute.utils` instead of being written again.

Shape conventions used across the package
-----------------------------------------
``X`` : 2-D float array of shape ``(n_samples, n_features)``.
    Rows are samples (axis 0) and columns are features (axis 1).
``y`` : 1-D array of shape ``(n_samples,)``.
    Labels may be integers or strings.
"""

from __future__ import annotations

import numbers

import numpy as np

# Reuse the Assignment 1 validators (written for NumCompute) rather than
# duplicating them. They are re-exported so users only need this package.
from numcompute.utils import (
    as_numeric_array,
    get_array_info,
    safe_divide,
    validate_array,
)

__all__ = [
    "NotFittedError",
    "as_numeric_array",
    "check_X",
    "check_X_y",
    "check_consistent_length",
    "check_is_fitted",
    "check_labels",
    "check_n_features",
    "check_positive_int",
    "check_random_state",
    "get_array_info",
    "safe_divide",
    "validate_array",
]


class NotFittedError(ValueError, AttributeError):
    """Raised when a model or transformer is used before it has learned anything.

    It inherits from both ``ValueError`` and ``AttributeError`` so code that
    catches either of those errors still works.
    """


def check_X(X, name="X", allow_nan=True, allow_empty=True):
    """Validate a feature matrix and return it as a 2-D float array.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Numeric feature data. Boolean and integer data are converted to
        float. A 1-D input is rejected because it is ambiguous (one sample
        or one feature?).
    name : str, default="X"
        Name used in error messages.
    allow_nan : bool, default=True
        If False, any NaN value raises an error.
    allow_empty : bool, default=True
        If True, an array with zero rows (an empty stream chunk) is
        accepted. An array with zero columns is never accepted.

    Returns
    -------
    numpy.ndarray of shape (n_samples, n_features), dtype float64
        A float copy of the data (the caller's array is never modified).

    Raises
    ------
    TypeError
        If the data are not numeric.
    ValueError
        If the array is not 2-D, has no columns, is empty when
        ``allow_empty`` is False, contains infinity, or contains NaN when
        ``allow_nan`` is False.
    """
    arr = np.asarray(X)
    if arr.dtype == np.bool_:
        # Booleans are valid features (for example one-hot columns).
        arr = arr.astype(np.float64)

    arr = as_numeric_array(arr, name=name)

    if arr.ndim == 1:
        raise ValueError(
            f"{name} must be 2-D with shape (n_samples, n_features); got a "
            f"1-D array of shape {arr.shape}. Use {name}.reshape(-1, 1) for "
            f"a single feature or {name}.reshape(1, -1) for a single sample."
        )
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D with shape (n_samples, n_features); "
            f"got {arr.ndim}-D array of shape {arr.shape}"
        )
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must have at least one feature column")
    if not allow_empty and arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")

    arr = arr.astype(np.float64, copy=True)

    if np.isinf(arr).any():
        raise ValueError(f"{name} must not contain infinite values")
    if not allow_nan and np.isnan(arr).any():
        raise ValueError(
            f"{name} contains NaN values; handle missing values first, "
            "for example with numcompute_stream.preprocessing.Imputer"
        )
    return arr


def check_labels(y, name="y"):
    """Validate class labels and return them as a 1-D array.

    Parameters
    ----------
    y : array-like of shape (n_samples,) or (n_samples, 1)
        Class labels (integers, booleans or strings). A column vector of
        shape ``(n_samples, 1)`` is flattened.
    name : str, default="y"
        Name used in error messages.

    Returns
    -------
    numpy.ndarray of shape (n_samples,)

    Raises
    ------
    ValueError
        If ``y`` has more than one column or contains NaN or None.
    """
    arr = np.asarray(y)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be 1-D; got a single scalar value")
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D with shape (n_samples,); got shape {arr.shape}"
        )
    if arr.dtype == object:
        if any(v is None for v in arr):
            raise ValueError(f"{name} must not contain None labels")
        try:
            # Object arrays of plain numbers or strings become a normal dtype.
            arr = np.asarray(arr.tolist())
        except (TypeError, ValueError):
            pass
    if np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any():
        raise ValueError(f"{name} must not contain NaN labels")
    return arr


def check_consistent_length(*arrays, names=None):
    """Check that all arrays have the same number of samples (rows).

    Parameters
    ----------
    *arrays : array-like
        Arrays whose first dimension must match.
    names : sequence of str, optional
        Names used in the error message.

    Raises
    ------
    ValueError
        If the first dimensions differ.
    """
    if names is None:
        names = [f"array_{i + 1}" for i in range(len(arrays))]
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) > 1:
        detail = ", ".join(f"{n}={k}" for n, k in zip(names, lengths))
        raise ValueError(f"Inputs have different numbers of samples: {detail}")


def check_X_y(X, y, allow_nan=True, allow_empty=True):
    """Validate a feature matrix and its labels together.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    allow_nan : bool, default=True
        Passed to :func:`check_X`.
    allow_empty : bool, default=True
        Passed to :func:`check_X`.

    Returns
    -------
    X : numpy.ndarray of shape (n_samples, n_features), dtype float64
    y : numpy.ndarray of shape (n_samples,)

    Raises
    ------
    ValueError
        If either input is invalid or their lengths differ.
    """
    X = check_X(X, allow_nan=allow_nan, allow_empty=allow_empty)
    y = check_labels(y)
    check_consistent_length(X, y, names=("X", "y"))
    return X, y


def check_n_features(X, n_features_expected, owner="this estimator"):
    """Check that ``X`` has the number of columns seen during fitting.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Already validated with :func:`check_X`.
    n_features_expected : int
        Number of columns the estimator was fitted on.
    owner : str
        Name of the estimator, used in the error message.

    Raises
    ------
    ValueError
        If the number of columns differs.
    """
    if X.shape[1] != n_features_expected:
        raise ValueError(
            f"X has {X.shape[1]} features, but {owner} was fitted with "
            f"{n_features_expected} features"
        )


def check_is_fitted(estimator, attributes, hint=None):
    """Raise :class:`NotFittedError` if the estimator has not been fitted.

    Parameters
    ----------
    estimator : object
        The estimator or transformer to check.
    attributes : str or sequence of str
        Attribute names that only exist (or are only non-None) after fitting.
    hint : str, optional
        Extra advice added to the error message.

    Raises
    ------
    NotFittedError
        If any listed attribute is missing or None.
    """
    if isinstance(attributes, str):
        attributes = [attributes]
    missing = [a for a in attributes if getattr(estimator, a, None) is None]
    if missing:
        message = (
            f"This {type(estimator).__name__} instance is not fitted yet. "
            "Call 'fit' or 'partial_fit' with some data first."
        )
        if hint:
            message += f" {hint}"
        raise NotFittedError(message)


def check_positive_int(value, name, minimum=1, allow_none=False):
    """Validate an integer setting such as ``buffer_size`` or ``max_depth``.

    Parameters
    ----------
    value : int or None
        The value to check. Booleans are rejected even though Python treats
        them as integers.
    name : str
        Parameter name used in the error message.
    minimum : int, default=1
        Smallest allowed value.
    allow_none : bool, default=False
        Whether ``None`` is allowed (for example ``max_depth=None``).

    Returns
    -------
    int or None

    Raises
    ------
    TypeError
        If the value is not an integer.
    ValueError
        If the value is smaller than ``minimum``.
    """
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, numbers.Integral
    ):
        raise TypeError(f"{name} must be an integer; got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}")
    return int(value)


def check_random_state(seed):
    """Turn ``seed`` into a :class:`numpy.random.Generator`.

    Parameters
    ----------
    seed : None, int or numpy.random.Generator
        ``None`` gives a fresh unpredictable generator, an integer gives a
        repeatable generator, and an existing generator is returned as is.

    Returns
    -------
    numpy.random.Generator

    Raises
    ------
    TypeError
        If ``seed`` is any other type.
    """
    is_int = isinstance(seed, numbers.Integral) and not isinstance(
        seed, (bool, np.bool_)
    )
    if seed is None or is_int:
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator):
        return seed
    raise TypeError(
        f"random_state must be None, an int or a numpy Generator; got {seed!r}"
    )
