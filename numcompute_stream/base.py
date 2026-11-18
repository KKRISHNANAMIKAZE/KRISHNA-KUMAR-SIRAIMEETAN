"""
Shared base classes for the streaming framework.

Every component in ``numcompute_stream`` follows one of three patterns, and
each pattern has a base class here:

* :class:`StreamingTransformer` - learns from chunks and changes data
  (scalers, imputers, encoders). Methods: ``partial_fit``, ``transform``.
* :class:`StreamingClassifier` - learns from chunks and predicts labels
  (the decision tree). Methods: ``partial_fit``, ``predict``,
  ``predict_proba``.
* :class:`StreamingMetric` - accumulates a score over chunks
  (accuracy, confusion matrix, ...). Methods: ``update``, ``result``.

Because all components share these interfaces, the pipeline and the stream
trainer can work with any of them without knowing which class they are.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod

import numpy as np

from .utils import check_X_y

__all__ = [
    "BaseStreamEstimator",
    "StreamingClassifier",
    "StreamingMetric",
    "StreamingTransformer",
]


class BaseStreamEstimator:
    """Parameter handling shared by every component.

    Subclasses must store each ``__init__`` argument as an attribute with the
    same name. Learned state is stored in attributes ending with ``_`` (for
    example ``mean_``), which keeps settings and learned values separate.
    """

    @classmethod
    def _param_names(cls):
        """Return the names of the ``__init__`` arguments, in order."""
        signature = inspect.signature(cls.__init__)
        return [
            name
            for name, p in signature.parameters.items()
            if name != "self"
            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]

    def get_params(self):
        """Return the component's settings as a dictionary.

        Returns
        -------
        dict
            Maps each ``__init__`` argument name to its current value.
        """
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params):
        """Change one or more settings.

        Learned state is not cleared; call ``reset()`` afterwards if the new
        settings should apply to a fresh start.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If a name is not one of the component's settings.
        """
        valid = set(self._param_names())
        for name, value in params.items():
            if name not in valid:
                raise ValueError(
                    f"Invalid parameter {name!r} for {type(self).__name__}. "
                    f"Valid parameters are: {sorted(valid)}"
                )
            setattr(self, name, value)
        return self

    def clone(self):
        """Return a new, unfitted copy with the same settings.

        Useful for comparing several copies of a model on the same stream
        (for example decision trees with different ``buffer_size`` values).
        """
        return type(self)(**self.get_params())

    def __repr__(self):
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"


class StreamingTransformer(BaseStreamEstimator, ABC):
    """Base class for components that learn from chunks and transform data.

    Subclasses implement :meth:`reset`, :meth:`partial_fit` and
    :meth:`transform`. The batch methods :meth:`fit` and
    :meth:`fit_transform` are built from those, so batch and streaming
    training always use the same code.
    """

    @abstractmethod
    def reset(self):
        """Forget everything learned so far. Returns ``self``."""

    @abstractmethod
    def partial_fit(self, X, y=None):
        """Update the learned statistics with one chunk. Returns ``self``."""

    @abstractmethod
    def transform(self, X):
        """Transform ``X`` using the statistics learned so far."""

    def fit(self, X, y=None):
        """Learn from ``X`` from scratch (``reset`` then ``partial_fit``).

        Returns
        -------
        self
        """
        self.reset()
        return self.partial_fit(X, y)

    def fit_transform(self, X, y=None):
        """Fit from scratch on ``X``, then return the transformed ``X``."""
        return self.fit(X, y).transform(X)

    def partial_fit_transform(self, X, y=None):
        """Update with one chunk, then return that chunk transformed."""
        return self.partial_fit(X, y).transform(X)


class StreamingClassifier(BaseStreamEstimator, ABC):
    """Base class for classifiers that learn from a stream of chunks.

    Subclasses implement :meth:`reset`, :meth:`partial_fit`, :meth:`predict`
    and :meth:`predict_proba`.
    """

    @abstractmethod
    def reset(self):
        """Forget everything learned so far. Returns ``self``."""

    @abstractmethod
    def partial_fit(self, X, y):
        """Learn from one chunk. Returns ``self``."""

    @abstractmethod
    def predict(self, X):
        """Return predicted labels, shape ``(n_samples,)``."""

    @abstractmethod
    def predict_proba(self, X):
        """Return class probabilities, shape ``(n_samples, n_classes)``."""

    def fit(self, X, y):
        """Learn from ``X, y`` from scratch (``reset`` then ``partial_fit``).

        Returns
        -------
        self
        """
        self.reset()
        return self.partial_fit(X, y)

    def score(self, X, y):
        """Return the accuracy of :meth:`predict` on ``X`` against ``y``.

        Returns
        -------
        float
            Fraction of correct predictions, between 0 and 1.

        Raises
        ------
        ValueError
            If ``X`` has no rows.
        """
        X, y = check_X_y(X, y, allow_empty=False)
        return float(np.mean(self.predict(X) == y))


class StreamingMetric(BaseStreamEstimator, ABC):
    """Base class for metrics that accumulate over a stream.

    Typical use::

        metric = Accuracy()
        for y_true, y_pred in chunks:
            metric.update(y_true, y_pred)
        print(metric.result())

    Subclasses implement :meth:`reset`, :meth:`update` and :meth:`result`.
    A metric that needs predicted probabilities instead of predicted labels
    (such as ROC AUC) sets ``needs_proba = True``.
    """

    #: True if ``update`` expects scores/probabilities rather than labels.
    needs_proba = False

    @abstractmethod
    def reset(self):
        """Clear all accumulated counts. Returns ``self``."""

    @abstractmethod
    def update(self, y_true, y_pred):
        """Add one chunk of true and predicted labels. Returns ``self``."""

    @abstractmethod
    def result(self):
        """Return the metric value over everything seen since the last reset."""
