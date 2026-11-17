"""
CSV loading and chunking for streaming experiments.

Three ways to get data into the framework:

* :func:`read_csv` - load a whole CSV file at once.
* :func:`iter_csv_chunks` - read a CSV file lazily, a few rows at a time.
  Only one chunk is held in memory, so this works for files larger than
  memory and is the closest thing to a real data stream.
* :func:`chunk_array` - split arrays that are already in memory into chunks
  (for example after :func:`read_csv`) to simulate a stream.

Only the Python standard library ``csv`` module and NumPy are used.
"""

from __future__ import annotations

import csv
from collections import namedtuple
from pathlib import Path

import numpy as np

from .utils import check_positive_int, check_random_state

__all__ = ["CSVData", "DEFAULT_MISSING", "chunk_array", "iter_csv_chunks", "read_csv"]

#: Cell values treated as missing. They are loaded as NaN in ``X``.
DEFAULT_MISSING = ("", "na", "nan", "null", "none", "?")

CSVData = namedtuple("CSVData", ["X", "y", "feature_names"])
CSVData.__doc__ = """Data loaded from a CSV file.

Attributes
----------
X : numpy.ndarray of shape (n_samples, n_features), dtype float64
    Feature values. Missing cells are NaN.
y : numpy.ndarray of shape (n_samples,) or None
    Labels from the label column, or None if no label column was given.
    Labels are integers if every label is a whole number, floats if they
    are all numbers, and strings otherwise.
feature_names : list of str
    One name per column of ``X``.
"""


# ----------------------------------------------------------------------
# Internal helpers shared by read_csv and iter_csv_chunks
# ----------------------------------------------------------------------

def _open_rows(path, delimiter, encoding):
    """Yield ``(line_number, row)`` for every non-blank row of the file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    # utf-8-sig removes the invisible byte-order mark Excel adds on Windows.
    with path.open("r", newline="", encoding=encoding) as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            if row and any(cell.strip() for cell in row):
                yield reader.line_num, [cell.strip() for cell in row]


def _resolve_layout(first_row, header, label_column):
    """Work out the column names and which column holds the labels.

    Returns
    -------
    n_columns : int
    label_index : int or None
    feature_names : list of str
    """
    n_columns = len(first_row)
    names = list(first_row) if header else [f"x{i}" for i in range(n_columns)]

    label_index = None
    if label_column is not None:
        if isinstance(label_column, str):
            if not header:
                raise ValueError(
                    "label_column can only be a column name when header=True"
                )
            if label_column not in names:
                raise ValueError(
                    f"label_column {label_column!r} not found in header {names}"
                )
            label_index = names.index(label_column)
        elif isinstance(label_column, (int, np.integer)) and not isinstance(
            label_column, bool
        ):
            if not -n_columns <= label_column < n_columns:
                raise ValueError(
                    f"label_column {label_column} is out of range for a file "
                    f"with {n_columns} columns"
                )
            label_index = int(label_column) % n_columns
        else:
            raise TypeError("label_column must be an int, a str or None")

    feature_names = [n for i, n in enumerate(names) if i != label_index]
    if not feature_names:
        raise ValueError("The file has no feature columns besides the label")
    if header:
        # Number feature names from 0 if they were blank in the header.
        feature_names = [n or f"x{i}" for i, n in enumerate(feature_names)]
    return n_columns, label_index, feature_names


def _convert_labels(labels):
    """Convert label strings to int, then float, otherwise keep strings."""
    for dtype in (np.int64, np.float64):
        try:
            return labels.astype(dtype)
        except ValueError:
            continue
    return labels


def _rows_to_data(rows, line_numbers, n_columns, label_index, feature_names,
                  missing):
    """Turn a list of string rows into a :class:`CSVData` (vectorised)."""
    lengths = np.fromiter((len(r) for r in rows), dtype=np.int64, count=len(rows))
    bad = np.flatnonzero(lengths != n_columns)
    if bad.size:
        i = bad[0]
        raise ValueError(
            f"Line {line_numbers[i]} has {lengths[i]} values; "
            f"expected {n_columns}"
        )

    n_features = len(feature_names)
    if not rows:
        y = np.empty(0) if label_index is not None else None
        return CSVData(np.empty((0, n_features)), y, list(feature_names))

    table = np.array(rows, dtype=str)

    y = None
    if label_index is not None:
        labels = table[:, label_index]
        is_missing = np.isin(np.char.lower(labels), missing)
        if is_missing.any():
            i = np.flatnonzero(is_missing)[0]
            raise ValueError(f"Line {line_numbers[i]} has a missing label")
        y = _convert_labels(labels)
        table = np.delete(table, label_index, axis=1)

    # Mark missing cells, then convert every column to float in one step.
    missing_mask = np.isin(np.char.lower(table), missing)
    table = np.where(missing_mask, "nan", table)
    try:
        X = table.astype(np.float64)
    except ValueError:
        # Slow path only used to build a helpful error message.
        for r, row in enumerate(table):
            for c, cell in enumerate(row):
                try:
                    float(cell)
                except ValueError:
                    raise ValueError(
                        f"Line {line_numbers[r]}, column "
                        f"{feature_names[c]!r}: cannot convert {cell!r} to a "
                        "number. Feature columns must be numeric; set "
                        "label_column if this is the label column."
                    ) from None
        raise
    return CSVData(X, y, list(feature_names))


def _normalise_missing(missing_values):
    return tuple(str(m).strip().lower() for m in missing_values)


# ----------------------------------------------------------------------
# Public functions
# ----------------------------------------------------------------------

def read_csv(path, delimiter=",", header=True, label_column=None,
             missing_values=DEFAULT_MISSING, encoding="utf-8-sig"):
    """Load a whole CSV file into a feature matrix and optional labels.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the CSV file.
    delimiter : str, default=","
        Column separator.
    header : bool, default=True
        Whether the first non-blank row holds column names. If False,
        features are named ``x0, x1, ...``.
    label_column : int, str or None, default=None
        Column holding the class labels, given by position (negative
        numbers count from the end) or by name. It is removed from ``X``
        and returned as ``y``.
    missing_values : sequence of str, default=DEFAULT_MISSING
        Cell values treated as missing (compared without case). Missing
        feature cells become NaN. A missing label raises an error.
    encoding : str, default="utf-8-sig"
        File encoding.

    Returns
    -------
    CSVData
        Named tuple ``(X, y, feature_names)``; it can be unpacked as
        ``X, y, names = read_csv(...)``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty, rows have different lengths, a feature value
        is not numeric, or a label is missing. The message gives the line
        number.

    Examples
    --------
    >>> X, y, names = read_csv("iris.csv", label_column="species")
    """
    missing = _normalise_missing(missing_values)
    rows_iter = _open_rows(path, delimiter, encoding)
    try:
        first_line, first_row = next(rows_iter)
    except StopIteration:
        raise ValueError(f"CSV file is empty: {path}") from None

    n_columns, label_index, names = _resolve_layout(
        first_row, header, label_column
    )
    rows, lines = ([], []) if header else ([first_row], [first_line])
    for line, row in rows_iter:
        rows.append(row)
        lines.append(line)
    return _rows_to_data(rows, lines, n_columns, label_index, names, missing)


def iter_csv_chunks(path, chunk_size, delimiter=",", header=True,
                    label_column=None, missing_values=DEFAULT_MISSING,
                    encoding="utf-8-sig"):
    """Read a CSV file lazily, yielding one chunk of rows at a time.

    Only ``chunk_size`` rows are held in memory at once, so this simulates a
    real data stream. The parameters are the same as :func:`read_csv`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the CSV file.
    chunk_size : int
        Number of rows per chunk (at least 1). The last chunk may be
        smaller.
    delimiter, header, label_column, missing_values, encoding
        See :func:`read_csv`.

    Yields
    ------
    CSVData
        ``(X_chunk, y_chunk, feature_names)`` for each chunk, with
        ``X_chunk`` of shape ``(chunk_rows, n_features)``.

    Raises
    ------
    FileNotFoundError, ValueError
        As for :func:`read_csv`. Errors in later rows are raised when that
        chunk is reached.

    Examples
    --------
    >>> for X_chunk, y_chunk, _ in iter_csv_chunks("data.csv", 200,
    ...                                            label_column=-1):
    ...     model.partial_fit(X_chunk, y_chunk)
    """
    chunk_size = check_positive_int(chunk_size, "chunk_size")
    missing = _normalise_missing(missing_values)
    rows_iter = _open_rows(path, delimiter, encoding)
    try:
        first_line, first_row = next(rows_iter)
    except StopIteration:
        raise ValueError(f"CSV file is empty: {path}") from None

    n_columns, label_index, names = _resolve_layout(
        first_row, header, label_column
    )
    rows, lines = ([], []) if header else ([first_row], [first_line])
    for line, row in rows_iter:
        if len(rows) == chunk_size:
            yield _rows_to_data(rows, lines, n_columns, label_index, names,
                                missing)
            rows, lines = [], []
        rows.append(row)
        lines.append(line)
    if rows:
        yield _rows_to_data(rows, lines, n_columns, label_index, names, missing)


def chunk_array(X, y=None, chunk_size=None, n_chunks=None, shuffle=False,
                random_state=None):
    """Split in-memory data into consecutive chunks to simulate a stream.

    Give exactly one of ``chunk_size`` or ``n_chunks``.

    Parameters
    ----------
    X : array-like of shape (n_samples, ...)
        Data to split along axis 0 (rows).
    y : array-like of shape (n_samples,), optional
        Labels split in the same way as ``X``.
    chunk_size : int, optional
        Rows per chunk. The last chunk may be smaller.
    n_chunks : int, optional
        Number of chunks. Chunk sizes differ by at most one row.
    shuffle : bool, default=False
        Shuffle the rows (with ``X`` and ``y`` kept together) before
        splitting. Leave False to keep the original order, which matters
        when the data drift over time.
    random_state : None, int or numpy.random.Generator
        Seed for shuffling.

    Yields
    ------
    tuple (X_chunk, y_chunk)
        ``y_chunk`` is None when ``y`` is None.

    Raises
    ------
    ValueError
        If both or neither of ``chunk_size`` and ``n_chunks`` are given,
        ``n_chunks`` is larger than the number of rows, or ``X`` and ``y``
        have different lengths.
    """
    if (chunk_size is None) == (n_chunks is None):
        raise ValueError("Give exactly one of chunk_size or n_chunks")

    X = np.asarray(X)
    if X.ndim == 0:
        raise ValueError("X must have at least one dimension")
    n = X.shape[0]
    if y is not None:
        y = np.asarray(y)
        if y.shape[0] != n:
            raise ValueError(
                f"X and y have different numbers of samples: {n} and "
                f"{y.shape[0]}"
            )

    if shuffle:
        order = check_random_state(random_state).permutation(n)
        X = X[order]
        y = y[order] if y is not None else None

    if chunk_size is not None:
        chunk_size = check_positive_int(chunk_size, "chunk_size")
        starts = np.arange(0, n, chunk_size)
        stops = np.minimum(starts + chunk_size, n)
    else:
        n_chunks = check_positive_int(n_chunks, "n_chunks")
        if n_chunks > n:
            raise ValueError(
                f"n_chunks ({n_chunks}) cannot be larger than the number "
                f"of samples ({n})"
            )
        bounds = np.linspace(0, n, n_chunks + 1).round().astype(int)
        starts, stops = bounds[:-1], bounds[1:]

    for start, stop in zip(starts, stops):
        yield X[start:stop], (y[start:stop] if y is not None else None)
