"""Tests for numcompute_stream.io (CSV loading and chunking)."""

import numpy as np
import pytest

from numcompute_stream.io import chunk_array, iter_csv_chunks, read_csv


def write(tmp_path, text, name="data.csv", encoding="utf-8"):
    path = tmp_path / name
    path.write_text(text, encoding=encoding)
    return path


IRIS_LIKE = (
    "sepal,petal,species\n"
    "5.1,1.4,setosa\n"
    "6.2,4.5,versicolor\n"
    "5.9,,virginica\n"
    "4.9,NA,setosa\n"
    "6.0,?,versicolor\n"
)


def test_read_csv_header_and_string_labels_by_name(tmp_path):
    X, y, names = read_csv(write(tmp_path, IRIS_LIKE), label_column="species")
    assert X.shape == (5, 2)
    assert names == ["sepal", "petal"]
    assert y.tolist() == ["setosa", "versicolor", "virginica", "setosa",
                          "versicolor"]


def test_read_csv_missing_cells_become_nan(tmp_path):
    X, _, _ = read_csv(write(tmp_path, IRIS_LIKE), label_column=-1)
    # Empty cell, "NA" and "?" are all treated as missing.
    assert np.isnan(X[2:, 1]).all()
    assert not np.isnan(X[:2]).any()


def test_read_csv_integer_labels_by_position(tmp_path):
    path = write(tmp_path, "label,a,b\n1,0.5,2\n0,1.5,3\n")
    X, y, names = read_csv(path, label_column=0)
    assert y.dtype == np.int64
    assert y.tolist() == [1, 0]
    assert names == ["a", "b"]
    assert X.tolist() == [[0.5, 2.0], [1.5, 3.0]]


def test_read_csv_without_header_or_labels(tmp_path):
    X, y, names = read_csv(write(tmp_path, "1,2\n3,4\n"), header=False)
    assert X.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert y is None
    assert names == ["x0", "x1"]


def test_read_csv_ignores_excel_bom_and_blank_lines(tmp_path):
    path = write(tmp_path, "a,b\n\n1,2\n   \n3,4\n", encoding="utf-8-sig")
    X, _, names = read_csv(path)
    assert names == ["a", "b"]          # no invisible character in "a"
    assert X.shape == (2, 2)


def test_read_csv_ragged_row_reports_line_number(tmp_path):
    path = write(tmp_path, "a,b\n1,2\n3\n")
    with pytest.raises(ValueError, match="Line 3 has 1 values; expected 2"):
        read_csv(path)


def test_read_csv_text_in_feature_column_reports_line_and_column(tmp_path):
    path = write(tmp_path, "a,b\n1,2\n3,abc\n")
    with pytest.raises(ValueError, match="Line 3, column 'b'.*'abc'"):
        read_csv(path)


def test_read_csv_missing_label_raises(tmp_path):
    path = write(tmp_path, "a,label\n1,0\n2,\n")
    with pytest.raises(ValueError, match="Line 3 has a missing label"):
        read_csv(path, label_column="label")


def test_read_csv_bad_label_column_arguments(tmp_path):
    path = write(tmp_path, "a,b\n1,2\n")
    with pytest.raises(ValueError, match="not found in header"):
        read_csv(path, label_column="c")
    with pytest.raises(ValueError, match="out of range"):
        read_csv(path, label_column=5)
    with pytest.raises(ValueError, match="header=True"):
        read_csv(path, header=False, label_column="a")


def test_read_csv_missing_or_empty_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_csv(tmp_path / "nope.csv")
    with pytest.raises(ValueError, match="empty"):
        read_csv(write(tmp_path, ""))


def test_iter_csv_chunks_match_whole_file(tmp_path):
    rows = "\n".join(f"{i},{i * 2},{i % 3}" for i in range(23))
    path = write(tmp_path, "a,b,label\n" + rows + "\n")
    chunks = list(iter_csv_chunks(path, chunk_size=5, label_column="label"))

    assert [len(c.X) for c in chunks] == [5, 5, 5, 5, 3]
    X_all, y_all, _ = read_csv(path, label_column="label")
    assert np.array_equal(np.vstack([c.X for c in chunks]), X_all)
    assert np.array_equal(np.concatenate([c.y for c in chunks]), y_all)


def test_iter_csv_chunks_reads_lazily(tmp_path):
    # Line 5 is broken, but the first chunk (lines 2-3) must still arrive.
    path = write(tmp_path, "a,b\n1,2\n3,4\n5,6\n7\n")
    stream = iter_csv_chunks(path, chunk_size=2)
    first = next(stream)
    assert first.X.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(ValueError, match="Line 5"):
        next(stream)


def test_chunk_array_by_size_keeps_order_and_last_chunk_smaller():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    chunks = list(chunk_array(X, y, chunk_size=4))
    assert [len(cx) for cx, _ in chunks] == [4, 4, 2]
    assert np.array_equal(np.concatenate([cy for _, cy in chunks]), y)


def test_chunk_array_by_count_is_balanced():
    chunks = list(chunk_array(np.zeros((10, 1)), n_chunks=3))
    sizes = [len(cx) for cx, cy in chunks]
    assert sum(sizes) == 10
    assert max(sizes) - min(sizes) <= 1
    assert all(cy is None for _, cy in chunks)


def test_chunk_array_shuffle_keeps_rows_paired_and_is_repeatable():
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10) * 10
    a = list(chunk_array(X, y, chunk_size=3, shuffle=True, random_state=1))
    b = list(chunk_array(X, y, chunk_size=3, shuffle=True, random_state=1))
    for (xa, ya), (xb, yb) in zip(a, b):
        assert np.array_equal(xa, xb)
        assert np.array_equal(ya, xa.ravel() * 10)   # pairs stay together
    order = np.concatenate([cx.ravel() for cx, _ in a])
    assert not np.array_equal(order, np.arange(10))


def test_chunk_array_argument_errors():
    X = np.zeros((5, 2))
    with pytest.raises(ValueError, match="exactly one"):
        list(chunk_array(X, chunk_size=2, n_chunks=2))
    with pytest.raises(ValueError, match="exactly one"):
        list(chunk_array(X))
    with pytest.raises(ValueError, match="cannot be larger"):
        list(chunk_array(X, n_chunks=6))
    with pytest.raises(ValueError, match="different numbers of samples"):
        list(chunk_array(X, np.zeros(4), chunk_size=2))
