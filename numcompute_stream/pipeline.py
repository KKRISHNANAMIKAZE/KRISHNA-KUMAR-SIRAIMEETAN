"""
Chain transformers and a final model into one streaming object.

A :class:`Pipeline` passes data through each step in order. For streaming,
``partial_fit`` updates every step with the chunk: each transformer first
learns from the chunk and then transforms it, and the transformed chunk is
passed to the next step. ``predict`` sends new data through the same steps
using what each has learned so far.

Example::

    pipe = Pipeline([
        ("impute", Imputer()),
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=5, buffer_size=500)),
    ])
    for X_chunk, y_chunk in stream:
        pipe.partial_fit(X_chunk, y_chunk)
    pipe.predict(X_new)

Design note: the tree buffers chunks *after* they have been scaled, and the
scaler's mean and variance keep changing, so rows in one buffer may have
been scaled slightly differently. The scaler's statistics settle quickly
(they are averages over everything seen), and a tree only compares values
within one feature, so the effect is small. Re-scaling raw rows at every
retrain would remove it but would mean keeping raw copies of the buffer.
"""

from __future__ import annotations

import numpy as np

from .base import BaseStreamEstimator

__all__ = ["Pipeline"]


class Pipeline(BaseStreamEstimator):
    """A chain of streaming transformers followed by a final step.

    Parameters
    ----------
    steps : list of (str, object) tuples
        Named steps in order. Every step except the last must have
        ``partial_fit`` and ``transform`` (a streaming transformer). The last
        step must have ``partial_fit``; it is usually a classifier.

    Raises
    ------
    ValueError
        If ``steps`` is empty, a name is repeated, or a step is missing a
        required method.
    """

    def __init__(self, steps):
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self):
        if not isinstance(self.steps, (list, tuple)) or not self.steps:
            raise ValueError("steps must be a non-empty list of (name, step) pairs")
        names = []
        for i, item in enumerate(self.steps):
            if not (isinstance(item, (list, tuple)) and len(item) == 2
                    and isinstance(item[0], str)):
                raise ValueError(f"Step {i} must be a (name, step) pair; got {item!r}")
            name, step = item
            if not hasattr(step, "partial_fit"):
                raise ValueError(f"Step {name!r} has no partial_fit method")
            is_last = i == len(self.steps) - 1
            if not is_last and not hasattr(step, "transform"):
                raise ValueError(
                    f"Step {name!r} is not the last step, so it must have a "
                    "transform method"
                )
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError(f"Step names must be unique; got {names}")

    # ------------------------------------------------------------------
    @property
    def named_steps(self):
        """Dictionary from step name to step."""
        return dict(self.steps)

    @property
    def final_step(self):
        """The last step (usually the model)."""
        return self.steps[-1][1]

    @property
    def classes_(self):
        """Class labels of the final step."""
        return getattr(self.final_step, "classes_", None)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.named_steps[key]
        return self.steps[key][1]

    def __len__(self):
        return len(self.steps)

    def clone(self):
        """A new, unfitted pipeline with cloned steps."""
        return Pipeline([(name, step.clone()) for name, step in self.steps])

    def reset(self):
        """Reset every step. Returns ``self``."""
        for _, step in self.steps:
            step.reset()
        return self

    # ------------------------------------------------------------------
    def _transformers(self):
        return [step for _, step in self.steps[:-1]]

    def _transform_inner(self, X):
        for step in self._transformers():
            X = step.transform(X)
        return X

    def partial_fit(self, X, y=None):
        """Update every step with one chunk.

        Each transformer learns from the chunk and then transforms it; the
        final step learns from the fully transformed chunk.

        Parameters
        ----------
        X : array-like of shape (n_chunk, n_features)
        y : array-like of shape (n_chunk,), optional

        Returns
        -------
        self
        """
        if np.asarray(X).shape[0] == 0:
            return self
        for step in self._transformers():
            X = step.partial_fit(X, y).transform(X)
        self.final_step.partial_fit(X, y)
        return self

    def fit(self, X, y=None):
        """Fit every step from scratch on the full data.

        Returns
        -------
        self
        """
        for step in self._transformers():
            X = step.fit(X, y).transform(X)
        self.final_step.fit(X, y)
        return self

    def flush(self):
        """Ask the final step to train on its buffer now, if it can."""
        if hasattr(self.final_step, "flush"):
            self.final_step.flush()
        return self

    def predict(self, X):
        """Transform ``X`` through the transformers, then predict."""
        return self.final_step.predict(self._transform_inner(X))

    def predict_proba(self, X):
        """Transform ``X`` through the transformers, then predict probabilities."""
        return self.final_step.predict_proba(self._transform_inner(X))

    def score(self, X, y):
        """Accuracy of the pipeline's predictions on ``X, y``."""
        return self.final_step.score(self._transform_inner(X), y)

    def transform(self, X):
        """Apply every step's ``transform`` (the last step must be a transformer)."""
        if not hasattr(self.final_step, "transform"):
            raise AttributeError(
                "The last step has no transform method; use predict instead"
            )
        return self.final_step.transform(self._transform_inner(X))
