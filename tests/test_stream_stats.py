"""Tests for numcompute_stream.stats (streaming statistics)."""

import numpy as np
import pytest

from numcompute_stream.io import chunk_array
from numcompute_stream.stats import StreamingHistogram, StreamingStats


def data_with_nan(n=500, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(5.0, 2.0, size=(n, d))
    X[rng.random(X.shape) < 0.1] = np.nan
    return X


def feed(stats, X, chunk_size):
    for chunk, _ in chunk_array(X, chunk_size=chunk_size):
        stats.update_stats(chunk)
    return stats


@pytest.mark.parametrize("ddof", [0, 1])
def test_streaming_moments_match_numpy_with_nan(ddof):
    X = data_with_nan()
    s = feed(StreamingStats(ddof=ddof), X, chunk_size=37)   # uneven chunks
    assert np.allclose(s.mean_, np.nanmean(X, axis=0))
    assert np.allclose(s.var_, np.nanvar(X, axis=0, ddof=ddof))
    assert np.allclose(s.min_, np.nanmin(X, axis=0))
    assert np.allclose(s.max_, np.nanmax(X, axis=0))
    assert np.array_equal(s.n_nan_, np.isnan(X).sum(axis=0))
    assert s.n_samples_seen_ == len(X)


def test_chunk_size_does_not_change_the_result():
    X = data_with_nan()
    one = feed(StreamingStats(), X, chunk_size=len(X))
    many = feed(StreamingStats(), X, chunk_size=1)
    assert np.allclose(one.mean_, many.mean_)
    assert np.allclose(one.var_, many.var_)


def test_variance_is_stable_with_a_huge_offset():
    # The naive "mean of squares minus square of mean" loses all precision
    # here; the Chan/Welford merge does not.
    rng = np.random.default_rng(1)
    X = 1e9 + rng.normal(size=(5000, 1))
    s = feed(StreamingStats(), X, chunk_size=10)
    assert s.var_[0] == pytest.approx(X.var(), rel=1e-6)


def test_all_nan_feature_gives_nan_statistics_not_errors():
    X = np.array([[1.0, np.nan], [3.0, np.nan]])
    result = StreamingStats().update_stats(X).result()
    assert result["count"].tolist() == [2, 0]
    assert np.isnan(result["mean"][1]) and np.isnan(result["var"][1])
    assert result["mean"][0] == 2.0


@pytest.mark.parametrize("method", ["linear", "lower", "higher", "nearest",
                                    "midpoint"])
def test_exact_quantiles_match_numpy_for_every_method(method):
    X = data_with_nan()
    s = feed(StreamingStats(quantile_method="exact"), X, chunk_size=50)
    q = [0.0, 0.1, 0.5, 0.93, 1.0]
    expected = np.nanquantile(X, q, axis=0, method=method)
    assert np.allclose(s.quantile(q, method=method), expected)


def test_reservoir_is_exact_while_stream_fits_in_capacity():
    X = data_with_nan(n=300)
    s = feed(StreamingStats(capacity=1000, random_state=0), X, chunk_size=40)
    assert np.allclose(s.median(), np.nanmedian(X, axis=0))


def test_reservoir_memory_is_bounded_and_estimate_is_close():
    X = data_with_nan(n=20000, d=2, seed=3)
    s = feed(StreamingStats(capacity=2000, random_state=0), X, chunk_size=500)
    assert s._sample.shape == (2000, 2)          # memory stays fixed
    error = np.abs(s.median() - np.nanmedian(X, axis=0))
    assert np.all(error < 0.15)                  # values have std 2


def test_reservoir_is_repeatable_with_a_seed():
    X = data_with_nan(n=3000)
    a = feed(StreamingStats(capacity=100, random_state=5), X, chunk_size=300)
    b = feed(StreamingStats(capacity=100, random_state=5), X, chunk_size=300)
    assert np.array_equal(a.quantile(0.3), b.quantile(0.3), equal_nan=True)


def test_quantile_argument_errors():
    s = StreamingStats().update_stats([[1.0], [2.0]])
    with pytest.raises(ValueError, match="between 0 and 1"):
        s.quantile(50)
    with pytest.raises(ValueError, match="method"):
        s.quantile(0.5, method="cubic")
    with pytest.raises(ValueError, match="disabled"):
        StreamingStats(quantile_method=None).update_stats([[1.0]]).quantile(0.5)
    with pytest.raises(ValueError, match="No data"):
        StreamingStats().quantile(0.5)


def test_feature_count_change_raises_and_empty_chunk_is_ignored():
    s = StreamingStats().update_stats(np.ones((2, 3)))
    s.update_stats(np.zeros((0, 3)))
    assert s.n_samples_seen_ == 2
    with pytest.raises(ValueError, match="X_chunk has 2 features"):
        s.update_stats(np.ones((2, 2)))


def test_histogram_over_stream_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 1000)
    h = StreamingHistogram(bins=8, value_range=(0, 10))
    for chunk, _ in chunk_array(x, chunk_size=90):
        h.update_stats(chunk)
    counts, edges = h.result()
    expected, _ = np.histogram(x, bins=8, range=(0, 10))
    assert np.array_equal(counts, expected)


def test_histogram_sliding_window_and_out_of_range_values():
    h = StreamingHistogram(bins=5, value_range=(0, 10), window=2)
    h.update_stats([1.0, 2.0, 3.0])
    h.update_stats([11.0, -1.0, np.nan, 5.0])
    h.update_stats([9.0, 9.0])
    counts, _ = h.result()
    assert counts.tolist() == [0, 0, 1, 0, 2]     # first chunk has left the window
    assert (h.n_below_, h.n_above_, h.n_nan_) == (1, 1, 1)


def test_histogram_takes_range_from_first_chunk():
    h = StreamingHistogram(bins=4).update_stats([2.0, 2.0])
    assert h.edges_[0] == 1.5 and h.edges_[-1] == 2.5   # widened, not zero-width
    with pytest.raises(ValueError, match="one feature"):
        h.update_stats(np.ones((3, 2)))
