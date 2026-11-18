"""
StreamTrainer: runs a model over a stream of chunks and logs what happens.

For each chunk the trainer uses **test-then-train** (prequential)
evaluation:

1. ``score_chunk`` - predict the new chunk with the current model and
   update the metrics. The model has not seen this chunk yet, so this is an
   honest measure of how it does on new data.
2. ``fit_chunk`` - pass the chunk to ``partial_fit`` so the model can
   learn from it.

Scoring after training would test the model on data it has just learned
from and would make the accuracy look better than it really is.

Every chunk adds one record to ``history_`` with the chunk accuracy,
cumulative accuracy, every metric, whether the tree retrained, timings and
the model's memory footprint.
"""

from __future__ import annotations

import time

import numpy as np

from .metrics import F1, Accuracy
from .utils import NotFittedError, check_positive_int, check_X_y, nbytes_of

__all__ = ["StreamTrainer"]


def _final_estimator(model):
    """The last step of a pipeline, or the model itself."""
    return getattr(model, "final_step", model)


class StreamTrainer:
    """Train a model chunk by chunk and log metrics over time.

    Parameters
    ----------
    model : object
        Anything with ``partial_fit`` and ``predict`` (a classifier or a
        :class:`~numcompute_stream.pipeline.Pipeline`). For metrics that
        need probabilities, it must also have ``predict_proba``.
    metrics : dict of str to StreamingMetric, optional
        Metrics to update with every scored chunk. Defaults to cumulative
        accuracy, rolling accuracy (last ``rolling_window`` chunks) and
        macro F1.
    rolling_window : int, default=5
        Window used by the default rolling accuracy.
    verbose : bool, default=False
        Print one line per chunk.

    Attributes
    ----------
    history_ : list of dict
        One record per chunk (see :meth:`process_chunk`).
    """

    def __init__(self, model, metrics=None, rolling_window=5, verbose=False):
        if not (hasattr(model, "partial_fit") and hasattr(model, "predict")):
            raise TypeError("model must have partial_fit and predict methods")
        self.model = model
        self.rolling_window = check_positive_int(rolling_window, "rolling_window")
        if metrics is None:
            metrics = {
                "accuracy": Accuracy(),
                "rolling_accuracy": Accuracy(window=rolling_window),
                "f1_macro": F1(average="macro"),
            }
        for name, metric in metrics.items():
            if not (hasattr(metric, "update") and hasattr(metric, "result")):
                raise TypeError(f"Metric {name!r} must have update and result")
        self.metrics = metrics
        self.verbose = verbose
        self.reset()

    def reset(self):
        """Clear the history and all metrics (the model is not reset)."""
        self.history_ = []
        self._n_correct = 0
        self._n_scored = 0
        self._n_seen = 0
        for metric in self.metrics.values():
            metric.reset()
        return self

    # ------------------------------------------------------------------
    def score_chunk(self, X, y):
        """Predict a chunk with the current model and update the metrics.

        Parameters
        ----------
        X : array-like of shape (n_chunk, n_features)
        y : array-like of shape (n_chunk,)

        Returns
        -------
        dict
            ``chunk_accuracy``, ``cumulative_accuracy``, ``predict_seconds``,
            ``scored`` (False while the model has not been trained yet) and
            the current value of every metric. Accuracy is NaN for an
            unscored chunk.
        """
        X, y = check_X_y(X, y)
        start = time.perf_counter()
        try:
            y_pred = self.model.predict(X)
        except NotFittedError:
            y_pred = None             # still filling the first buffer
        record = {"predict_seconds": time.perf_counter() - start,
                  "scored": y_pred is not None}

        if y_pred is None or len(y) == 0:
            record["chunk_accuracy"] = float("nan")
        else:
            correct = int(np.sum(y_pred == y))
            self._n_correct += correct
            self._n_scored += len(y)
            record["chunk_accuracy"] = correct / len(y)
            proba = None
            for metric in self.metrics.values():
                if getattr(metric, "needs_proba", False):
                    if proba is None:
                        proba = self._positive_scores(X, metric)
                    metric.update(y, proba)
                else:
                    metric.update(y, y_pred)
        record["cumulative_accuracy"] = (
            self._n_correct / self._n_scored if self._n_scored else float("nan")
        )
        for name, metric in self.metrics.items():
            record[name] = metric.result()
        return record

    def _positive_scores(self, X, metric):
        """Probability of the metric's positive class for each row."""
        proba = self.model.predict_proba(X)
        classes = np.asarray(self.model.classes_)
        hit = np.flatnonzero(classes == metric.pos_label)
        return proba[:, hit[0]] if hit.size else np.zeros(len(X))

    def fit_chunk(self, X, y):
        """Let the model learn from a chunk and time it.

        Returns
        -------
        dict
            ``fit_seconds``, ``retrained`` (True if the tree was rebuilt
            during this call), ``n_retrains``, ``buffer_count`` and
            ``memory_bytes`` (NumPy memory held by the model).
        """
        final = _final_estimator(self.model)
        before = getattr(final, "n_retrains_", 0)
        start = time.perf_counter()
        self.model.partial_fit(X, y)
        seconds = time.perf_counter() - start
        after = getattr(final, "n_retrains_", 0)
        self._n_seen += len(y)
        return {
            "fit_seconds": seconds,
            "retrained": after > before,
            "n_retrains": after,
            "buffer_count": getattr(final, "buffer_count_", None),
            "memory_bytes": nbytes_of(self.model),
        }

    def process_chunk(self, X, y):
        """Test-then-train on one chunk and append a record to ``history_``.

        Returns
        -------
        dict
            The record: ``chunk`` (1-based index), ``n_samples``,
            ``samples_seen`` and everything from :meth:`score_chunk` and
            :meth:`fit_chunk`.
        """
        X, y = check_X_y(X, y)
        record = {"chunk": len(self.history_) + 1, "n_samples": len(y)}
        record.update(self.score_chunk(X, y))
        record.update(self.fit_chunk(X, y))
        record["samples_seen"] = self._n_seen
        self.history_.append(record)
        if self.verbose:
            print(self.format_record(record))
        return record

    def run(self, chunks, max_chunks=None):
        """Process every chunk from an iterable.

        Parameters
        ----------
        chunks : iterable
            Yields ``(X, y)`` pairs, or ``(X, y, feature_names)`` from
            :func:`~numcompute_stream.io.iter_csv_chunks`.
        max_chunks : int, optional
            Stop after this many chunks.

        Returns
        -------
        list of dict
            ``history_``.
        """
        for i, chunk in enumerate(chunks):
            if max_chunks is not None and i >= max_chunks:
                break
            self.process_chunk(chunk[0], chunk[1])
        return self.history_

    # ------------------------------------------------------------------
    @staticmethod
    def format_record(record):
        """One readable log line for a record."""
        def fmt(value):
            return "  -  " if value != value else f"{value:.3f}"   # NaN check
        flag = "  RETRAINED" if record["retrained"] else ""
        return (
            f"chunk {record['chunk']:>3} | seen {record['samples_seen']:>6} | "
            f"chunk acc {fmt(record['chunk_accuracy'])} | "
            f"cumulative acc {fmt(record['cumulative_accuracy'])} | "
            f"buffer {record['buffer_count']} | "
            f"mem {record['memory_bytes'] / 1024:.1f} KB{flag}"
        )

    def get(self, key):
        """One logged value for every chunk, as an array (for plotting).

        Example: ``trainer.get("cumulative_accuracy")``.
        """
        if not self.history_:
            return np.array([])
        if key not in self.history_[0]:
            raise KeyError(f"{key!r} is not logged; available: "
                           f"{sorted(self.history_[0])}")
        return np.array([r[key] for r in self.history_])

    def retrain_chunks(self):
        """1-based chunk numbers at which the model was retrained."""
        return [r["chunk"] for r in self.history_ if r["retrained"]]

    def summary(self):
        """Totals for the whole run, as a dictionary."""
        if not self.history_:
            return {}
        last = self.history_[-1]
        return {
            "chunks": len(self.history_),
            "samples_seen": last["samples_seen"],
            "final_cumulative_accuracy": last["cumulative_accuracy"],
            "retrains": last["n_retrains"],
            "total_fit_seconds": float(np.sum(self.get("fit_seconds"))),
            "peak_memory_bytes": int(np.max(self.get("memory_bytes"))),
            **{name: metric.result() for name, metric in self.metrics.items()},
        }
