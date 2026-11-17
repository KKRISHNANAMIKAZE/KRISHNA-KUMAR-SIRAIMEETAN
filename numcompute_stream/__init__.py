"""
numcompute_stream
=================

A streaming machine-learning framework built on NumPy, extending the
NumCompute toolkit. Data arrive in chunks, every component learns
incrementally through ``partial_fit`` / ``update``, and a buffered decision
tree retrains once enough data has been collected.

Modules
-------
utils    Shared input validation and error messages.
base     Base classes that every transformer, classifier and metric follows.
io       CSV loading and chunking to simulate a data stream.
"""

from .base import (
    BaseStreamEstimator,
    StreamingClassifier,
    StreamingMetric,
    StreamingTransformer,
)
from .io import CSVData, chunk_array, iter_csv_chunks, read_csv
from .utils import NotFittedError

__version__ = "0.1.0"

__all__ = [
    "BaseStreamEstimator",
    "CSVData",
    "NotFittedError",
    "StreamingClassifier",
    "StreamingMetric",
    "StreamingTransformer",
    "chunk_array",
    "iter_csv_chunks",
    "read_csv",
]
