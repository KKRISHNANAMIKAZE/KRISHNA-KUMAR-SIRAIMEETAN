"""
Streaming preprocessing: transformers that learn one chunk at a time.

Every class follows :class:`~numcompute_stream.base.StreamingTransformer`:
``partial_fit`` updates what has been learned, ``transform`` uses what has
been learned so far, and ``fit`` / ``fit_transform`` do the same in one go.

* :class:`StandardScaler` - running mean and variance (Chan / Welford
  merge via :class:`~numcompute_stream.stats.StreamingStats`).
* :class:`Imputer` - fills NaN with the running mean, running median or a
  constant.
* :class:`OneHotEncoder` - adds new category columns as new categories
  appear in the stream.
"""

from __future__ import annotations

import numpy as np

from .base import StreamingTransformer
from .stats import StreamingStats
from .utils import (
    check_is_fitted,
    check_n_features,
    check_positive_int,
    check_X,
)

__all__ = ["Imputer", "OneHotEncoder", "StandardScaler"]


class StandardScaler(StreamingTransformer):
    """Scale features to mean 0 and standard deviation 1, learned online.

    Parameters
    ----------
    with_mean : bool, default=True
        Subtract the running mean.
    with_std : bool, default=True
        Divide by the running standard deviation.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
    var_ : ndarray of shape (n_features,)
        Population variance (divide by n) over everything seen.
    scale_ : ndarray of shape (n_features,)
        Standard deviation used for dividing. A feature with zero variance
        (constant so far) gets scale 1, so it becomes 0 instead of NaN.
    n_samples_seen_ : int
    n_features_in_ : int

    Notes
    -----
    NaN values are skipped when learning and stay NaN after
    ``transform``, so an :class:`Imputer` can be placed before or after the
    scaler.
    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.reset()

    def reset(self):
        """Forget the learned mean and variance. Returns ``self``."""
        self._stats = StreamingStats(quantile_method=None)
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = 0
        return self

    def partial_fit(self, X, y=None):
        """Update the running mean and variance with one chunk.

        Parameters
        ----------
        X : array-like of shape (n_chunk, n_features)
        y : ignored

        Returns
        -------
        self
        """
        X = check_X(X)
        if self.n_features_in_ is not None:
            check_n_features(X, self.n_features_in_, owner=type(self).__name__)
        if X.shape[0] == 0:
            return self
        self._stats.update_stats(X)
        self.n_features_in_ = self._stats.n_features_in_
        self.n_samples_seen_ = self._stats.n_samples_seen_
        seen = self._stats.count_ > 0
        self.mean_ = np.where(seen, self._stats.mean_, 0.0)
        self.var_ = np.where(seen, self._stats.var_, 0.0)
        std = np.sqrt(self.var_)
        self.scale_ = np.where(std > 0, std, 1.0)
        return self

    def _check_input(self, X):
        check_is_fitted(self, "mean_")
        X = check_X(X)
        check_n_features(X, self.n_features_in_, owner=type(self).__name__)
        return X

    def transform(self, X):
        """Scale ``X`` with the statistics learned so far.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
        """
        X = self._check_input(X)
        if self.with_mean:
            X = X - self.mean_
        if self.with_std:
            X = X / self.scale_
        return X

    def inverse_transform(self, X):
        """Undo :meth:`transform`."""
        X = self._check_input(X)
        if self.with_std:
            X = X * self.scale_
        if self.with_mean:
            X = X + self.mean_
        return X


class Imputer(StreamingTransformer):
    """Fill missing values (NaN) with estimates that update over the stream.

    Parameters
    ----------
    strategy : {"mean", "median", "constant"}, default="mean"
        "mean" and "median" use the values seen so far in each feature
        (the median is approximate once more than ``capacity`` rows have
        been seen; see :class:`~numcompute_stream.stats.StreamingStats`).
        "constant" always uses ``fill_value``.
    fill_value : float, default=0.0
        Used by "constant", and for any feature that has had no non-NaN
        value yet.
    capacity : int, default=10000
        Reservoir size for the "median" strategy.
    random_state : None, int or numpy.random.Generator
        Seed for the reservoir sample.

    Attributes
    ----------
    statistics_ : ndarray of shape (n_features,)
        The value each feature's NaN entries are replaced with.
    n_features_in_ : int
    """

    def __init__(self, strategy="mean", fill_value=0.0, capacity=10000,
                 random_state=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.capacity = capacity
        self.random_state = random_state
        if strategy not in ("mean", "median", "constant"):
            raise ValueError(
                "strategy must be 'mean', 'median' or 'constant'; "
                f"got {strategy!r}"
            )
        check_positive_int(capacity, "capacity")
        self.reset()

    def reset(self):
        """Forget the learned fill values. Returns ``self``."""
        method = "reservoir" if self.strategy == "median" else None
        self._stats = StreamingStats(quantile_method=method,
                                     capacity=self.capacity,
                                     random_state=self.random_state)
        self.statistics_ = None
        self.n_features_in_ = None
        return self

    def partial_fit(self, X, y=None):
        """Update the fill values with one chunk.

        Returns
        -------
        self
        """
        X = check_X(X)
        if self.n_features_in_ is not None:
            check_n_features(X, self.n_features_in_, owner=type(self).__name__)
        if X.shape[0] == 0:
            return self
        self._stats.update_stats(X)
        self.n_features_in_ = X.shape[1]
        if self.strategy == "constant":
            values = np.full(X.shape[1], float(self.fill_value))
        elif self.strategy == "mean":
            values = self._stats.mean_
        else:
            values = self._stats.median()
        seen = self._stats.count_ > 0
        self.statistics_ = np.where(seen, values, float(self.fill_value))
        return self

    def transform(self, X):
        """Replace NaN in each feature with ``statistics_``.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            A copy with no NaN values.
        """
        check_is_fitted(self, "statistics_")
        X = check_X(X)
        check_n_features(X, self.n_features_in_, owner=type(self).__name__)
        return np.where(np.isnan(X), self.statistics_, X)


def _check_categorical(X):
    """Validate a 2-D array of categories (text or numbers)."""
    arr = np.asarray(X)
    if arr.ndim == 1:
        raise ValueError(
            "X must be 2-D with shape (n_samples, n_features); got a 1-D "
            "array. Use X.reshape(-1, 1) for a single column."
        )
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {arr.shape}")
    if arr.dtype == object:
        if any(v is None for v in arr.ravel()):
            raise ValueError("X must not contain None; fill missing values first")
        arr = arr.astype(str)
    if arr.dtype.kind == "f" and np.isnan(arr).any():
        raise ValueError("X must not contain NaN; fill missing values first")
    return arr


class OneHotEncoder(StreamingTransformer):
    """One-hot encode categorical columns, adding categories as they appear.

    Parameters
    ----------
    categories : "auto" or list of array-like, default="auto"
        "auto" learns categories from the data and adds new ones in later
        chunks. A list fixes the categories of each column up front.
    handle_unknown : {"error", "ignore"}, default="error"
        What :meth:`transform` does with a category it has not learned:
        raise an error, or output all zeros for that column.

    Attributes
    ----------
    categories_ : list of ndarray
        Categories of each input column, in the order they were learned.
    n_features_in_ : int
    n_features_out_ : int

    Notes
    -----
    **Column order is stable.** Output columns for the first chunk are
    ordered by input column, then by sorted category. A category first seen
    in a later chunk gets a *new column at the right-hand end* of the
    output, so earlier columns never move. A model trained on the old
    columns therefore keeps reading the right features; see
    ``DecisionTreeClassifier(allow_new_features=True)``, which pads its
    buffer with zeros for the new columns.
    """

    def __init__(self, categories="auto", handle_unknown="error"):
        self.categories = categories
        self.handle_unknown = handle_unknown
        if handle_unknown not in ("error", "ignore"):
            raise ValueError(
                f"handle_unknown must be 'error' or 'ignore'; got {handle_unknown!r}"
            )
        if not (isinstance(categories, str) and categories == "auto"):
            if not isinstance(categories, (list, tuple)):
                raise TypeError("categories must be 'auto' or a list of lists")
        self.reset()

    def reset(self):
        """Forget all learned categories. Returns ``self``."""
        self._cats = None        # categories per input column
        self._out_cols = None    # output column of each category
        self.n_features_in_ = None
        self.n_features_out_ = 0
        return self

    @property
    def categories_(self):
        return None if self._cats is None else [c.copy() for c in self._cats]

    def _start(self, n_columns):
        self.n_features_in_ = n_columns
        if isinstance(self.categories, str):          # "auto"
            self._cats = [np.array([]) for _ in range(n_columns)]
            self._out_cols = [np.array([], dtype=np.intp)
                              for _ in range(n_columns)]
            return
        if len(self.categories) != n_columns:
            raise ValueError(
                f"categories has {len(self.categories)} lists but X has "
                f"{n_columns} columns"
            )
        self._cats, self._out_cols = [], []
        for cats in self.categories:
            cats = np.asarray(cats)
            if len(np.unique(cats)) != len(cats):
                raise ValueError("categories must not contain duplicates")
            start = self.n_features_out_
            self._cats.append(cats)
            self._out_cols.append(np.arange(start, start + len(cats)))
            self.n_features_out_ += len(cats)

    def partial_fit(self, X, y=None):
        """Learn any new categories in this chunk.

        New categories get new output columns at the right-hand end.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            With fixed ``categories``, if the chunk has an unknown
            category and ``handle_unknown="error"``.
        """
        X = _check_categorical(X)
        if self._cats is None:
            self._start(X.shape[1])
        else:
            check_n_features(X, self.n_features_in_, owner=type(self).__name__)
        if X.shape[0] == 0:
            return self

        fixed = not isinstance(self.categories, str)
        for j in range(X.shape[1]):     # loop over columns, not rows
            seen = np.unique(X[:, j])
            known = self._cats[j]
            if known.size and (seen.dtype.kind in "US") != (known.dtype.kind in "US"):
                raise TypeError(
                    f"Column {j} changed type between chunks (text vs numbers)"
                )
            new = seen[~np.isin(seen, known)] if known.size else seen
            if new.size == 0:
                continue
            if fixed:
                if self.handle_unknown == "error":
                    raise ValueError(
                        f"Column {j} has unknown categories {new.tolist()}"
                    )
                continue
            start = self.n_features_out_
            self._cats[j] = np.concatenate([known, new]) if known.size else new
            self._out_cols[j] = np.concatenate(
                [self._out_cols[j], np.arange(start, start + new.size)]
            )
            self.n_features_out_ += new.size
        return self

    def transform(self, X):
        """One-hot encode ``X`` with the categories learned so far.

        Returns
        -------
        ndarray of shape (n_samples, n_features_out_), dtype float64
        """
        check_is_fitted(self, "_cats")
        X = _check_categorical(X)
        check_n_features(X, self.n_features_in_, owner=type(self).__name__)
        out = np.zeros((X.shape[0], self.n_features_out_))
        rows = np.arange(X.shape[0])
        for j in range(X.shape[1]):
            cats = self._cats[j]
            if cats.size == 0:
                found = np.zeros(X.shape[0], dtype=bool)
                where = found.astype(np.intp)
            else:
                order = np.argsort(cats)
                sorted_cats = cats[order]
                pos = np.clip(np.searchsorted(sorted_cats, X[:, j]), 0,
                              cats.size - 1)
                found = sorted_cats[pos] == X[:, j]
                where = self._out_cols[j][order][pos]
            if not found.all() and self.handle_unknown == "error":
                unknown = np.unique(X[~found, j]).tolist()
                raise ValueError(
                    f"Column {j} has categories not seen during fitting: "
                    f"{unknown}. Use handle_unknown='ignore' to encode them as "
                    "all zeros."
                )
            out[rows[found], where[found]] = 1.0
        return out

    def get_feature_names_out(self, input_features=None):
        """Names of the output columns, like ``"colour=red"``.

        Parameters
        ----------
        input_features : list of str, optional
            Input column names. Defaults to ``x0, x1, ...``.

        Returns
        -------
        list of str
        """
        check_is_fitted(self, "_cats")
        if input_features is None:
            input_features = [f"x{j}" for j in range(self.n_features_in_)]
        names = [""] * self.n_features_out_
        for j, (cats, cols) in enumerate(zip(self._cats, self._out_cols)):
            for cat, col in zip(cats, cols):
                names[col] = f"{input_features[j]}={cat}"
        return names
