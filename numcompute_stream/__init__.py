"""
numcompute_stream
=================

A streaming machine-learning framework built on NumPy, extending the
NumCompute toolkit. Data arrive in chunks, every component learns
incrementally through ``partial_fit`` / ``update`` / ``update_stats``, and
a buffered decision tree retrains once enough data has been collected.

Modules
-------
utils          Shared input validation and error messages.
base           Base classes that every transformer, classifier and metric follows.
io             CSV loading and chunking to simulate a data stream.
tree           DecisionTreeClassifier with buffered retraining.
stats          Streaming mean, variance, quantiles and histograms.
preprocessing  StandardScaler, Imputer and OneHotEncoder with partial_fit.
metrics        Streaming accuracy, precision, recall, F1, confusion matrix, AUC.
pipeline       Pipeline that chains transformers and a model.
stream         StreamTrainer: test-then-train loop with logging.
visualise      Plots for metrics over time, model comparison and predictions.

Quick start
-----------
>>> from numcompute_stream import (Pipeline, StandardScaler, Imputer,
...                                DecisionTreeClassifier, StreamTrainer,
...                                iter_csv_chunks)
>>> pipe = Pipeline([("impute", Imputer()), ("scale", StandardScaler()),
...                  ("model", DecisionTreeClassifier(max_depth=5,
...                                                   buffer_size=500))])
>>> trainer = StreamTrainer(pipe, verbose=True)
>>> trainer.run(iter_csv_chunks("data.csv", 250, label_column=-1))
"""

from .base import (
    BaseStreamEstimator,
    StreamingClassifier,
    StreamingMetric,
    StreamingTransformer,
)
from .io import CSVData, chunk_array, iter_csv_chunks, read_csv
from .metrics import (
    ROCAUC,
    F1,
    Accuracy,
    ConfusionMatrix,
    Precision,
    Recall,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from .pipeline import Pipeline
from .preprocessing import Imputer, OneHotEncoder, StandardScaler
from .stats import StreamingHistogram, StreamingStats
from .stream import StreamTrainer
from .tree import DecisionTreeClassifier
from .utils import NotFittedError

__version__ = "0.1.0"

__all__ = [
    "Accuracy",
    "BaseStreamEstimator",
    "CSVData",
    "ConfusionMatrix",
    "DecisionTreeClassifier",
    "F1",
    "Imputer",
    "NotFittedError",
    "OneHotEncoder",
    "Pipeline",
    "Precision",
    "ROCAUC",
    "Recall",
    "StandardScaler",
    "StreamTrainer",
    "StreamingClassifier",
    "StreamingHistogram",
    "StreamingMetric",
    "StreamingStats",
    "StreamingTransformer",
    "accuracy_score",
    "chunk_array",
    "confusion_matrix",
    "f1_score",
    "iter_csv_chunks",
    "precision_score",
    "read_csv",
    "recall_score",
    "roc_auc_score",
]
