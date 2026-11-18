"""
Streaming statistics: summaries that update one chunk at a time.

A stream never shows all the data at once, so these classes keep a small
running summary and update it with each new chunk through
``update_stats(X_chunk)``:

* :class:`StreamingStats` - per-feature count, mean, variance, standard
  deviation, min, max, NaN count and quantiles.
* :class:`StreamingHistogram` - histogram counts with fixed bins, over the
  whole stream or over a sliding window of recent chunks.

Mean and variance (exact, any chunk sizes)
------------------------------------------
Each chunk's own mean and sum of squared deviations (``M2``) are merged
into the running totals with Chan et al.'s parallel formula, the chunk
version of Welford's algorithm::

    n     = n_a + n_b
    delta = mean_b - mean_a
    mean  = mean_a + delta * n_b / n
    M2    = M2_a + M2_b + delta**2 * n_a * n_b / n

This gives the same result as computing over all the data at once, and
avoids the cancellation error of the "sum of squares minus square of sum"
shortcut. NaN values are skipped separately for each feature.

Quantiles (exact or approximate - chosen explicitly)
---------------------------------------------------
Quantiles cannot be updated exactly from a small summary, so there are two
documented options:

* ``quantile_method="exact"`` keeps every row. Results match
  ``np.nanquantile`` exactly, but memory grows with the stream.
* ``quantile_method="reservoir"`` (default) keeps a uniform random sample of
  at most ``capacity`` rows (reservoir sampling, Algorithm R). Memory stays
  fixed and the result is an approximation. While fewer than ``capacity``
  rows have been seen, the sample *is* all the data, so it is exact.
* ``quantile_method=None`` stores nothing, for when quantiles are not needed.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .base import BaseStreamEstimator
from .utils import (
    as_numeric_array,
    check_n_features,
    check_positive_int,
    check_random_state,
    check_X,
)

__all__ = ["StreamingHistogram", "StreamingStats", "quantile_from_sample"]

_QUANTILE_METHODS = ("linear", "lower", "higher", "nearest", "midpoint")


def quantile_from_sample(data, q, method="linear"):
    """Per-column quantiles of a 2-D array, ignoring NaN (vectorised).

    Parameters
    ----------
    data : ndarray of shape (n_rows, n_features)
    q : float or array-like of floats in [0, 1]
        Quantile(s) to compute, for example 0.5 for the median.
    method : {"linear", "lower", "higher", "nearest", "midpoint"}
        How to pick a value when the quantile falls between two data
        points ``a <= b`` at sorted positions ``i`` and ``i + 1``:
        "linear" interpolates, "lower" takes ``a``, "higher" takes ``b``,
        "nearest" takes the closer one (halfway goes to the even position,
        as in NumPy) and "midpoint" takes ``(a + b) / 2``.

    Returns
    -------
    ndarray of shape (n_features,) if ``q`` is a scalar, otherwise
    (len(q), n_features)
        NaN for a feature with no non-NaN values.

    Raises
    ------
    ValueError
        If ``q`` is outside [0, 1] or ``method`` is unknown.
    """
    if method not in _QUANTILE_METHODS:
        raise ValueError(f"method must be one of {_QUANTILE_METHODS}; got {method!r}")
    q_arr = np.asarray(q, dtype=float)
    if np.any((q_arr < 0) | (q_arr > 1)) or np.isnan(q_arr).any():
        raise ValueError("q must be between 0 and 1")
    q_2d = np.atleast_1d(q_arr)[:, None]                     # (n_q, 1)

    ordered = np.sort(data, axis=0)                          # NaN go last
    n_valid = np.sum(~np.isnan(data), axis=0)                # (n_features,)
    position = q_2d * (n_valid - 1)                          # (n_q, n_features)
    position = np.clip(position, 0, None)
    low = np.floor(position).astype(np.intp)
    high = np.ceil(position).astype(np.intp)
    if method == "nearest":
        low = high = np.around(position).astype(np.intp)     # half to even

    if ordered.shape[0] == 0:
        result = np.full(position.shape, np.nan)
    else:
        v_low = np.take_along_axis(ordered, low, axis=0)
        v_high = np.take_along_axis(ordered, high, axis=0)
        if method in ("linear",):
            result = v_low + (position - low) * (v_high - v_low)
        elif method == "higher":
            result = v_high
        elif method == "midpoint":
            result = (v_low + v_high) / 2.0
        else:                                                # lower, nearest
            result = v_low
        result = np.where(n_valid > 0, result, np.nan)
    return result[0] if q_arr.ndim == 0 else result


class StreamingStats(BaseStreamEstimator):
    """Running per-feature statistics, updated one chunk at a time.

    Parameters
    ----------
    ddof : int, default=0
        Delta degrees of freedom for the variance: 0 for the population
        variance (divide by n), 1 for the sample variance (divide by n - 1).
    quantile_method : {"reservoir", "exact"} or None, default="reservoir"
        How to support :meth:`quantile` (see the module notes).
    capacity : int, default=10000
        Maximum rows kept when ``quantile_method="reservoir"``.
    random_state : None, int or numpy.random.Generator
        Seed for reservoir sampling.

    Attributes
    ----------
    count_ : ndarray of shape (n_features,)
        Non-NaN values seen per feature.
    mean_, var_, std_, min_, max_ : ndarray of shape (n_features,)
        Running statistics (NaN for a feature with too few values).
    n_nan_ : ndarray of shape (n_features,)
        NaN values seen per feature.
    n_samples_seen_ : int
        Rows seen (including rows with NaN).
    n_features_in_ : int
        Number of features.

    Examples
    --------
    >>> stats = StreamingStats()
    >>> for X_chunk, _ in chunk_array(X, chunk_size=100):
    ...     stats.update_stats(X_chunk)
    >>> stats.mean_, stats.std_, stats.quantile(0.5)
    """

    def __init__(self, ddof=0, quantile_method="reservoir", capacity=10000,
                 random_state=None):
        self.ddof = ddof
        self.quantile_method = quantile_method
        self.capacity = capacity
        self.random_state = random_state
        self._validate_params()
        self.reset()

    def _validate_params(self):
        check_positive_int(self.ddof, "ddof", minimum=0)
        if self.quantile_method not in ("reservoir", "exact", None):
            raise ValueError(
                "quantile_method must be 'reservoir', 'exact' or None; "
                f"got {self.quantile_method!r}"
            )
        check_positive_int(self.capacity, "capacity")

    def reset(self):
        """Forget everything seen so far. Returns ``self``."""
        self.n_features_in_ = None
        self.n_samples_seen_ = 0
        self.count_ = None
        self.mean_ = None
        self._m2 = None
        self.min_ = None
        self.max_ = None
        self.n_nan_ = None
        self._sample = None
        self._rng = check_random_state(self.random_state)
        return self

    # ------------------------------------------------------------------
    def update_stats(self, X_chunk):
        """Update every statistic with one chunk.

        Parameters
        ----------
        X_chunk : array-like of shape (n_chunk, n_features)
            NaN values are skipped per feature. An empty chunk is ignored.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the number of features differs from earlier chunks.
        """
        X = check_X(X_chunk, name="X_chunk")
        if X.shape[0] == 0:
            return self
        if self.n_features_in_ is None:
            d = X.shape[1]
            self.n_features_in_ = d
            self.count_ = np.zeros(d, dtype=np.int64)
            self.mean_ = np.zeros(d)
            self._m2 = np.zeros(d)
            self.min_ = np.full(d, np.inf)
            self.max_ = np.full(d, -np.inf)
            self.n_nan_ = np.zeros(d, dtype=np.int64)
        else:
            check_n_features(X, self.n_features_in_, owner=type(self).__name__,
                             name="X_chunk")

        self._merge_moments(X)
        self._update_sample(X)
        self.n_samples_seen_ += X.shape[0]
        return self

    #: Alias so the class can be used like the streaming metrics.
    update = update_stats

    def _merge_moments(self, X):
        """Merge the chunk's count, mean and M2 into the running totals."""
        valid = ~np.isnan(X)
        n_b = valid.sum(axis=0)
        filled = np.where(valid, X, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_b = np.where(n_b > 0, filled.sum(axis=0) / n_b, 0.0)
        m2_b = np.sum(np.where(valid, X - mean_b, 0.0) ** 2, axis=0)

        n_a = self.count_
        n = n_a + n_b
        safe_n = np.where(n > 0, n, 1)
        delta = mean_b - self.mean_
        self.mean_ = np.where(n > 0, self.mean_ + delta * n_b / safe_n, 0.0)
        self._m2 = self._m2 + m2_b + delta ** 2 * n_a * n_b / safe_n
        self.count_ = n

        self.min_ = np.minimum(self.min_, np.where(valid, X, np.inf).min(axis=0))
        self.max_ = np.maximum(self.max_, np.where(valid, X, -np.inf).max(axis=0))
        self.n_nan_ = self.n_nan_ + (~valid).sum(axis=0)

    def _update_sample(self, X):
        """Keep rows for quantiles: all of them, or a reservoir sample."""
        if self.quantile_method is None:
            return
        if self._sample is None:
            self._sample = np.empty((0, X.shape[1]))
        if self.quantile_method == "exact":
            self._sample = np.vstack([self._sample, X])
            return

        # Reservoir sampling (Algorithm R), vectorised for a whole chunk.
        free = self.capacity - self._sample.shape[0]
        take = min(free, X.shape[0])
        if take > 0:
            self._sample = np.vstack([self._sample, X[:take]])
        rest = X[take:]
        if rest.shape[0] == 0:
            return
        # Row i of the stream (0-based) replaces slot j ~ Uniform{0..i}
        # if j < capacity. When several rows pick the same slot, the last
        # one wins - exactly what the one-at-a-time algorithm would do.
        index = self.n_samples_seen_ + take + np.arange(rest.shape[0])
        slot = self._rng.integers(0, index + 1)
        keep = slot < self.capacity
        rows, slots = rest[keep], slot[keep]
        if slots.size:
            unique_slots, first_from_end = np.unique(slots[::-1],
                                                     return_index=True)
            last = slots.size - 1 - first_from_end
            self._sample[unique_slots] = rows[last]

    # ------------------------------------------------------------------
    def _check_seen(self):
        if self.n_features_in_ is None:
            raise ValueError("No data seen yet; call update_stats first")

    @property
    def var_(self):
        """Variance per feature (NaN where count <= ddof)."""
        if self.count_ is None:
            return None
        denom = self.count_ - self.ddof
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(denom > 0, self._m2 / np.where(denom > 0, denom, 1),
                            np.nan)

    @property
    def std_(self):
        """Standard deviation per feature."""
        var = self.var_
        return None if var is None else np.sqrt(var)

    def quantile(self, q, method="linear"):
        """Quantile(s) per feature from the stored rows.

        Parameters
        ----------
        q : float or array-like in [0, 1]
        method : {"linear", "lower", "higher", "nearest", "midpoint"}
            See :func:`quantile_from_sample`.

        Returns
        -------
        ndarray of shape (n_features,) or (len(q), n_features)

        Raises
        ------
        ValueError
            If no data has been seen or ``quantile_method`` is None.
        """
        self._check_seen()
        if self.quantile_method is None:
            raise ValueError(
                "Quantiles are disabled (quantile_method=None); create the "
                "StreamingStats with quantile_method='reservoir' or 'exact'"
            )
        return quantile_from_sample(self._sample, q, method=method)

    def median(self):
        """Median per feature (``quantile(0.5)``)."""
        return self.quantile(0.5)

    def result(self):
        """All statistics in one dictionary of arrays."""
        self._check_seen()
        nothing = self.count_ == 0
        return {
            "count": self.count_.copy(),
            "mean": np.where(nothing, np.nan, self.mean_),
            "var": self.var_,
            "std": self.std_,
            "min": np.where(nothing, np.nan, self.min_),
            "max": np.where(nothing, np.nan, self.max_),
            "n_nan": self.n_nan_.copy(),
        }


class StreamingHistogram(BaseStreamEstimator):
    """Histogram of one feature, built chunk by chunk.

    Parameters
    ----------
    bins : int, default=10
        Number of equal-width bins.
    value_range : tuple (low, high) or None, default=None
        Lower and upper edge of the bins. If None, the range of the first
        non-empty chunk is used (widened by 0.5 each side if all its values
        are equal). Bins never move after that, so counts from different
        chunks can be added together.
    window : int or None, default=None
        If set, only the last ``window`` chunks are counted (a sliding
        window), so the histogram follows recent data.

    Attributes
    ----------
    edges_ : ndarray of shape (bins + 1,)
    counts_ : ndarray of shape (bins,)
        Counts per bin over the stream (or over the window).
    n_below_, n_above_ : int
        Values outside the bin range. They are counted here instead of
        being dropped without notice.
    n_nan_ : int
        NaN values skipped.
    """

    def __init__(self, bins=10, value_range=None, window=None):
        self.bins = bins
        self.value_range = value_range
        self.window = window
        check_positive_int(bins, "bins")
        check_positive_int(window, "window", allow_none=True)
        if value_range is not None:
            low, high = value_range
            if not low < high:
                raise ValueError("value_range must be (low, high) with low < high")
        self.reset()

    def reset(self):
        """Forget all counts (and the bins, if they came from data)."""
        self.edges_ = (None if self.value_range is None
                       else np.linspace(*self.value_range, self.bins + 1))
        self._chunks = deque(maxlen=self.window)
        self.n_below_ = 0
        self.n_above_ = 0
        self.n_nan_ = 0
        return self

    @property
    def counts_(self):
        if not self._chunks:
            return np.zeros(self.bins, dtype=np.int64)
        return np.sum(self._chunks, axis=0)

    def update_stats(self, values):
        """Add one chunk of values to the histogram.

        Parameters
        ----------
        values : array-like of shape (n,) or (n, 1)
            Values of a single feature. NaN values are skipped and counted.

        Returns
        -------
        self
        """
        x = as_numeric_array(values, name="values").astype(float)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.ravel()
        if x.ndim != 1:
            raise ValueError(
                "StreamingHistogram takes one feature: a 1-D array or a "
                f"single column; got shape {x.shape}"
            )
        is_nan = np.isnan(x)
        self.n_nan_ += int(is_nan.sum())
        x = x[~is_nan]
        if x.size == 0:
            return self
        if self.edges_ is None:
            low, high = float(x.min()), float(x.max())
            if low == high:
                low, high = low - 0.5, high + 0.5
            self.edges_ = np.linspace(low, high, self.bins + 1)

        below = x < self.edges_[0]
        above = x > self.edges_[-1]
        self.n_below_ += int(below.sum())
        self.n_above_ += int(above.sum())
        counts, _ = np.histogram(x[~below & ~above], bins=self.edges_)
        self._chunks.append(counts)
        return self

    update = update_stats

    def result(self):
        """Return ``(counts, edges)``."""
        return self.counts_, self.edges_
