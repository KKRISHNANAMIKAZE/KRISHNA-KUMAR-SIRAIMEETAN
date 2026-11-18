"""
Buffered decision tree classifier for streaming data.

The tree itself is a normal depth-limited CART-style classification tree,
built from scratch with NumPy. Streaming support comes from **buffered
retraining**: each call to :meth:`DecisionTreeClassifier.partial_fit` adds
the chunk to a buffer, and once the buffer holds at least ``buffer_size``
samples the tree is rebuilt on the buffered data and the buffer is cleared.
Between retrains, :meth:`~DecisionTreeClassifier.predict` uses the most
recently trained tree.

How a tree is built (same rules as the course guide)
----------------------------------------------------
A node stops and becomes a leaf when, checked in this order:

1. it is at ``max_depth``,
2. all its labels are the same (it is pure),
3. it has fewer than ``min_samples_split`` samples, or
4. no question lowers the impurity (the best gain is not above 0).

Otherwise it asks the question ``X[:, f] <= t`` with the biggest
information gain, sends the "yes" rows to the left child and the "no" rows
to the right, and builds the left child first. The course guide only uses
yes/no (0/1) features; this tree also handles real-valued features by
trying a threshold between every pair of neighbouring distinct values.

Tie rules (all deterministic)
-----------------------------
* Gains are rounded to 12 decimal places, so two splits whose gains differ
  only by floating-point noise count as a tie.
* A split must beat the best gain so far *strictly* (best gain starts at 0),
  so among tied splits the lowest feature index wins, then the lowest
  threshold.
* A leaf predicts the class with the most samples; a tie goes to the class
  that comes first in ``classes_`` (like ``np.bincount(y).argmax()``).

Two implementations
-------------------
``implementation="vectorized"`` (default) finds the best split for every
feature and threshold at once: it sorts each feature, takes a running total
(``cumsum``) of one-hot class counts so the left and right class counts for
*all* thresholds come out of one operation, then computes every impurity in
one step. ``implementation="loop"`` checks one threshold at a time in
Python, like a textbook. Both build exactly the same tree; the loop version
exists for the benchmark.
"""

from __future__ import annotations

import time
from collections import namedtuple

import numpy as np

from .base import StreamingClassifier
from .utils import (
    check_is_fitted,
    check_n_features,
    check_positive_int,
    check_random_state,
    check_X,
    check_X_y,
)

__all__ = ["DecisionTreeClassifier", "Tree"]

# Gains are rounded to this many decimals before comparing (see tie rules).
_GAIN_DECIMALS = 12
# Upper limit on the size of the temporary (rows x features x classes)
# array in the vectorised split search. Features are processed in blocks
# that stay under this, so memory use is bounded for large buffers.
_MAX_BLOCK_ELEMENTS = 4_000_000

Tree = namedtuple(
    "Tree",
    ["feature", "threshold", "left", "right", "value", "n_samples",
     "impurity", "depth"],
)
Tree.__doc__ = """A trained tree stored as flat arrays, one entry per node.

Node 0 is the root. For node ``i``:

feature[i] : int
    Feature the node asks about, or -1 for a leaf.
threshold[i] : float
    The question is ``X[:, feature[i]] <= threshold[i]`` (NaN for a leaf).
left[i], right[i] : int
    Child node for "yes" and "no" answers (-1 for a leaf).
value[i] : ndarray of shape (n_classes,)
    Number of training samples of each class that reached the node.
n_samples[i] : int
    Number of training samples that reached the node.
impurity[i] : float
    Gini or entropy of the node.
depth[i] : int
    Depth of the node (the root has depth 0).
"""


# ----------------------------------------------------------------------
# Impurity and split search
# ----------------------------------------------------------------------

def impurity(counts, n, criterion="gini"):
    """Gini impurity or entropy from class counts (vectorised).

    Parameters
    ----------
    counts : ndarray of shape (..., n_classes)
        Class counts. Any number of leading dimensions is allowed, so the
        impurity of many candidate splits is computed in one call.
    n : ndarray of shape (...) or float
        Total count for each row of ``counts`` (must be > 0).
    criterion : {"gini", "entropy"}

    Returns
    -------
    ndarray of shape (...)
        Gini: ``1 - sum(p**2)``. Entropy: ``-sum(p * log2(p))`` with
        ``0 * log2(0)`` taken as 0.
    """
    p = counts / np.asarray(n, dtype=float)[..., None]
    if criterion == "gini":
        return 1.0 - np.sum(p * p, axis=-1)
    log_p = np.log2(p, out=np.zeros_like(p), where=p > 0)
    return -np.sum(p * log_p, axis=-1)


def _midpoint(low, high):
    """Threshold halfway between two neighbouring distinct values.

    If the two floats are so close that the midpoint rounds up to ``high``,
    use ``low`` so that ``low`` still goes left and ``high`` goes right.
    """
    mid = (low + high) / 2.0
    return np.where(mid < high, mid, low)


def _best_split_vectorized(X, y, n_classes, features, criterion):
    """Find the best ``X[:, f] <= t`` split with NumPy array operations.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
        Class indices 0..n_classes-1.
    n_classes : int
    features : ndarray of int
        Features to consider, in increasing order.
    criterion : {"gini", "entropy"}

    Returns
    -------
    gain : float
        Information gain of the best split (0.0 if none helps).
    feature : int
        Best feature, or -1 if no split has gain above 0.
    threshold : float
        Best threshold (NaN if no split).
    """
    n = X.shape[0]
    parent_counts = np.bincount(y, minlength=n_classes).astype(float)
    parent_impurity = impurity(parent_counts, float(n), criterion)
    one_hot = np.eye(n_classes)[y]                    # (n, k)
    n_left = np.arange(1, n, dtype=float)[:, None]    # (n-1, 1)
    n_right = n - n_left

    best_gain, best_feature, best_threshold = 0.0, -1, np.nan
    block = max(1, _MAX_BLOCK_ELEMENTS // max(1, n * n_classes))
    for start in range(0, len(features), block):
        feats = features[start:start + block]
        values = X[:, feats]                                   # (n, m)
        order = np.argsort(values, axis=0, kind="stable")
        sorted_values = np.take_along_axis(values, order, axis=0)

        # Left counts for every threshold position, all features at once.
        left = np.cumsum(one_hot[order], axis=0)[:-1]          # (n-1, m, k)
        right = parent_counts - left
        weighted = (n_left * impurity(left, n_left, criterion)
                    + n_right * impurity(right, n_right, criterion)) / n
        gains = np.round(parent_impurity - weighted, _GAIN_DECIMALS)

        # Only split between two different values.
        valid = sorted_values[1:] > sorted_values[:-1]         # (n-1, m)
        gains = np.where(valid, gains, -np.inf)

        # Feature-major order so ties go to the lowest feature, then the
        # lowest threshold; argmax returns the first maximum.
        flat = gains.T.ravel()
        i = int(np.argmax(flat))
        if flat[i] > best_gain:
            f_pos, row = divmod(i, n - 1)
            best_gain = float(flat[i])
            best_feature = int(feats[f_pos])
            best_threshold = float(_midpoint(sorted_values[row, f_pos],
                                             sorted_values[row + 1, f_pos]))
    return best_gain, best_feature, best_threshold


def _best_split_loop(X, y, n_classes, features, criterion):
    """Find the best split by checking one threshold at a time.

    Same inputs, outputs and tie rules as :func:`_best_split_vectorized`.
    Kept for the loop-versus-vectorised benchmark and as a readable
    reference.
    """
    n = X.shape[0]
    parent_counts = np.bincount(y, minlength=n_classes).astype(float)
    parent_impurity = impurity(parent_counts, float(n), criterion)

    best_gain, best_feature, best_threshold = 0.0, -1, np.nan
    for f in features:
        column = X[:, f]
        distinct = np.unique(column)
        for low, high in zip(distinct[:-1], distinct[1:]):
            threshold = float(_midpoint(low, high))
            goes_left = column <= threshold
            n_l = float(goes_left.sum())
            n_r = n - n_l
            left = np.bincount(y[goes_left], minlength=n_classes).astype(float)
            right = parent_counts - left
            weighted = (n_l * impurity(left, n_l, criterion)
                        + n_r * impurity(right, n_r, criterion)) / n
            gain = float(np.round(parent_impurity - weighted, _GAIN_DECIMALS))
            if gain > best_gain:
                best_gain, best_feature, best_threshold = gain, int(f), threshold
    return best_gain, best_feature, best_threshold


_SPLITTERS = {"vectorized": _best_split_vectorized, "loop": _best_split_loop}


def _label_kind(labels):
    """'text' for string labels, 'number' for numeric or boolean labels."""
    return "text" if labels.dtype.kind in "USO" else "number"


# ----------------------------------------------------------------------
# The classifier
# ----------------------------------------------------------------------

class DecisionTreeClassifier(StreamingClassifier):
    """Depth-limited decision tree with buffered retraining for streams.

    Parameters
    ----------
    max_depth : int or None, default=5
        Maximum depth of the tree (the root has depth 0, so ``max_depth=1``
        means one question). None means grow until the other stopping rules
        apply.
    min_samples_split : int, default=2
        A node with fewer samples than this becomes a leaf.
    max_features : None, int, float, "sqrt" or "log2", default=None
        Number of features to try at each node. None tries all. An int is a
        count, a float in (0, 1] is a fraction, "sqrt" and "log2" are
        functions of the number of features. A random subset is drawn at
        each node using ``random_state``.
    criterion : {"gini", "entropy"}, default="gini"
        Impurity measure used to score splits.
    buffer_size : int, default=500
        :meth:`partial_fit` retrains the tree once the buffer holds at least
        this many samples, then clears the buffer.
    implementation : {"vectorized", "loop"}, default="vectorized"
        How to search for splits and route samples at prediction time. Both
        give identical results; "loop" is for benchmarking.
    allow_new_features : bool, default=False
        If False (the safe default), every chunk must have the same number
        of columns. If True, a later chunk may have *extra* columns on the
        right (for example new one-hot categories): buffered rows are padded
        with zeros, and the current tree ignores the extra columns until it
        is retrained.
    random_state : None, int or numpy.random.Generator, default=None
        Seed for the feature subsets when ``max_features`` is set.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Every class label seen so far, sorted.
    n_features_in_ : int
        Number of features the current tree was trained on.
    tree_ : Tree
        The current tree as flat arrays (see :class:`Tree`). None before
        the first training.
    n_retrains_ : int
        Number of times a tree has been built.
    retrain_log_ : list of dict
        One entry per build with keys ``n_samples`` (training size),
        ``seconds`` (build time) and ``samples_seen`` (stream position).
    n_samples_seen_ : int
        Total samples passed to ``fit`` and ``partial_fit``.

    Examples
    --------
    >>> tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
    >>> for X_chunk, y_chunk in stream:
    ...     tree.partial_fit(X_chunk, y_chunk)
    >>> tree.predict(X_new)
    """

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None,
                 criterion="gini", buffer_size=500, implementation="vectorized",
                 allow_new_features=False, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.buffer_size = buffer_size
        self.implementation = implementation
        self.allow_new_features = allow_new_features
        self.random_state = random_state
        self._validate_params()
        self.reset()

    # ------------------------------------------------------------------
    # Settings and state
    # ------------------------------------------------------------------

    def _validate_params(self):
        """Check every setting and raise a clear error if one is invalid."""
        check_positive_int(self.max_depth, "max_depth", allow_none=True)
        check_positive_int(self.min_samples_split, "min_samples_split",
                           minimum=2)
        check_positive_int(self.buffer_size, "buffer_size")
        if self.criterion not in ("gini", "entropy"):
            raise ValueError(
                f"criterion must be 'gini' or 'entropy'; got {self.criterion!r}"
            )
        if self.implementation not in _SPLITTERS:
            raise ValueError(
                "implementation must be 'vectorized' or 'loop'; "
                f"got {self.implementation!r}"
            )
        mf = self.max_features
        if not (mf is None or mf in ("sqrt", "log2")):
            if isinstance(mf, (bool, np.bool_)):
                raise TypeError("max_features must not be a boolean")
            if isinstance(mf, (int, np.integer)):
                check_positive_int(mf, "max_features")
            elif isinstance(mf, (float, np.floating)):
                if not 0.0 < mf <= 1.0:
                    raise ValueError(
                        f"max_features as a fraction must be in (0, 1]; got {mf}"
                    )
            else:
                raise TypeError(
                    "max_features must be None, an int, a float, 'sqrt' or "
                    f"'log2'; got {mf!r}"
                )
        check_random_state(self.random_state)

    def reset(self):
        """Forget the tree, the buffer and all seen classes.

        Returns
        -------
        self
        """
        self.classes_ = None
        self.n_features_in_ = None
        self.tree_ = None
        self.n_retrains_ = 0
        self.retrain_log_ = []
        self.n_samples_seen_ = 0
        self._tree_classes = None
        self._buffer_X = []
        self._buffer_y = []
        self._width = None
        self._rng = check_random_state(self.random_state)
        return self

    @property
    def buffer_count_(self):
        """Number of samples currently waiting in the buffer."""
        return int(sum(len(chunk) for chunk in self._buffer_y))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Train a new tree from scratch on ``X, y`` straight away.

        Unlike :meth:`partial_fit`, this ignores ``buffer_size``: the tree
        is built immediately and anything in the buffer is discarded.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Must not contain NaN (use an Imputer first).
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        self._validate_params()
        X, y = check_X_y(X, y, allow_nan=False, allow_empty=False)
        self.reset()
        self._update_classes(y)
        self._width = X.shape[1]
        self.n_samples_seen_ = len(y)
        self._train(X, y)
        return self

    def partial_fit(self, X, y):
        """Add a chunk to the buffer and retrain if the buffer is full.

        Parameters
        ----------
        X : array-like of shape (n_chunk, n_features)
            Must not contain NaN. An empty chunk (0 rows) is ignored.
        y : array-like of shape (n_chunk,)

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the chunk has NaN, a different number of features than
            earlier chunks, or a length mismatch between ``X`` and ``y``.
        TypeError
            If the labels change type (for example from numbers to text).
        """
        self._validate_params()
        X, y = check_X_y(X, y, allow_nan=False)
        if len(y) == 0:
            return self
        X = self._match_width(X)
        self._update_classes(y)
        self._buffer_X.append(X)
        self._buffer_y.append(y)
        self.n_samples_seen_ += len(y)
        if self.buffer_count_ >= self.buffer_size:
            self._retrain_from_buffer()
        return self

    def flush(self):
        """Retrain now on whatever is in the buffer, even if it is not full.

        Useful at the end of a stream, or to get a first tree quickly.
        Does nothing if the buffer is empty.

        Returns
        -------
        self
        """
        if self._buffer_y:
            self._retrain_from_buffer()
        return self

    def _match_width(self, X):
        """Check the chunk's column count against earlier chunks."""
        if self._width is None:
            self._width = X.shape[1]
        elif X.shape[1] > self._width and self.allow_new_features:
            extra = X.shape[1] - self._width
            # Older rows did not have the new columns, so 0 is their value.
            self._buffer_X = [np.pad(b, ((0, 0), (0, extra)))
                              for b in self._buffer_X]
            self._width = X.shape[1]
        elif X.shape[1] != self._width:
            check_n_features(X, self._width, owner=type(self).__name__)
        return X

    def _update_classes(self, y):
        """Add any new labels to ``classes_``."""
        new = np.unique(y)
        if self.classes_ is None:
            self.classes_ = new
            return
        if _label_kind(new) != _label_kind(self.classes_):
            raise TypeError(
                f"Labels changed type: earlier labels were "
                f"{_label_kind(self.classes_)}s ({self.classes_[:3].tolist()}), "
                f"this chunk has {_label_kind(new)}s ({new[:3].tolist()})"
            )
        self.classes_ = np.union1d(self.classes_, new)

    def _retrain_from_buffer(self):
        X = np.vstack(self._buffer_X)
        y = np.concatenate(self._buffer_y)
        self._buffer_X, self._buffer_y = [], []
        self._train(X, y)

    def _n_features_to_try(self, n_features):
        mf = self.max_features
        if mf is None:
            return n_features
        if mf == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(mf, (float, np.floating)):
            return max(1, int(mf * n_features))
        return min(int(mf), n_features)

    def _train(self, X, y):
        """Build a tree on ``X, y`` and record how long it took."""
        start = time.perf_counter()
        y_index = np.searchsorted(self.classes_, y)
        self.tree_ = self._build(X, y_index, len(self.classes_))
        self._tree_classes = self.classes_.copy()
        self.n_features_in_ = X.shape[1]
        self.n_retrains_ += 1
        self.retrain_log_.append({
            "n_samples": len(y),
            "seconds": time.perf_counter() - start,
            "samples_seen": self.n_samples_seen_,
        })

    def _build(self, X, y, n_classes):
        """Grow the tree with an explicit stack (left child first)."""
        find_split = _SPLITTERS[self.implementation]
        n_try = self._n_features_to_try(X.shape[1])
        all_features = np.arange(X.shape[1])

        feature, threshold, left, right = [], [], [], []
        value, n_samples, node_impurity, depth = [], [], [], []

        # Each stack entry: (row indices, depth, parent node, side).
        stack = [(np.arange(len(y)), 0, -1, "left")]
        while stack:
            rows, d, parent, side = stack.pop()
            node = len(feature)
            if parent >= 0:
                (left if side == "left" else right)[parent] = node

            counts = np.bincount(y[rows], minlength=n_classes).astype(float)
            feature.append(-1)
            threshold.append(np.nan)
            left.append(-1)
            right.append(-1)
            value.append(counts)
            n_samples.append(len(rows))
            node_impurity.append(float(impurity(counts, len(rows),
                                                self.criterion)))
            depth.append(d)

            # Stopping rules, in the same order as the course guide.
            if self.max_depth is not None and d >= self.max_depth:
                continue
            if np.count_nonzero(counts) <= 1:
                continue
            if len(rows) < self.min_samples_split:
                continue

            if n_try < X.shape[1]:
                candidates = np.sort(self._rng.choice(X.shape[1], n_try,
                                                      replace=False))
            else:
                candidates = all_features
            gain, f, t = find_split(X[rows], y[rows], n_classes, candidates,
                                    self.criterion)
            if f < 0:
                continue          # no question lowers the impurity

            feature[node], threshold[node] = f, t
            goes_left = X[rows, f] <= t
            # Pushed right first so the left ("yes") child is built first.
            stack.append((rows[~goes_left], d + 1, node, "right"))
            stack.append((rows[goes_left], d + 1, node, "left"))

        return Tree(
            feature=np.array(feature, dtype=np.intp),
            threshold=np.array(threshold, dtype=float),
            left=np.array(left, dtype=np.intp),
            right=np.array(right, dtype=np.intp),
            value=np.array(value, dtype=float).reshape(-1, n_classes),
            n_samples=np.array(n_samples, dtype=np.intp),
            impurity=np.array(node_impurity, dtype=float),
            depth=np.array(depth, dtype=np.intp),
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if self.tree_ is None:
            hint = (
                f"The buffer holds {self.buffer_count_} of {self.buffer_size} "
                "samples needed for the first tree; call flush() to train "
                "on them now."
            )
            check_is_fitted(self, "tree_", hint=hint)

    def _prepare_X(self, X):
        X = check_X(X, allow_nan=False)
        if X.shape[1] > self.n_features_in_ and self.allow_new_features:
            return X[:, :self.n_features_in_]
        check_n_features(X, self.n_features_in_, owner=type(self).__name__)
        return X

    def apply(self, X):
        """Return the index of the leaf each sample ends up in.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
            Leaf node index for each row.
        """
        self._check_fitted()
        X = self._prepare_X(X)
        if self.implementation == "loop":
            return self._apply_loop(X)
        return self._apply_vectorized(X)

    def _apply_vectorized(self, X):
        """Move all samples down the tree one level at a time.

        The Python loop runs once per tree level (at most ``max_depth``
        times), never once per sample.
        """
        tree = self.tree_
        node = np.zeros(X.shape[0], dtype=np.intp)
        active = np.flatnonzero(tree.feature[node] >= 0)
        while active.size:
            current = node[active]
            goes_left = X[active, tree.feature[current]] <= tree.threshold[current]
            node[active] = np.where(goes_left, tree.left[current],
                                    tree.right[current])
            active = active[tree.feature[node[active]] >= 0]
        return node

    def _apply_loop(self, X):
        """Walk each sample down the tree separately (benchmark reference)."""
        tree = self.tree_
        out = np.empty(X.shape[0], dtype=np.intp)
        for i, row in enumerate(X):
            node = 0
            while tree.feature[node] >= 0:
                if row[tree.feature[node]] <= tree.threshold[node]:
                    node = tree.left[node]
                else:
                    node = tree.right[node]
            out[i] = node
        return out

    def predict_proba(self, X):
        """Class probabilities from the leaf each sample reaches.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Columns follow ``classes_``. A class first seen after the
            current tree was trained gets probability 0 until the next
            retrain.

        Raises
        ------
        NotFittedError
            If no tree has been trained yet.
        """
        counts = self._leaf_counts(X)
        proba = counts / counts.sum(axis=1, keepdims=True)
        if len(self.classes_) == len(self._tree_classes):
            return proba
        full = np.zeros((proba.shape[0], len(self.classes_)))
        full[:, np.searchsorted(self.classes_, self._tree_classes)] = proba
        return full

    def _leaf_counts(self, X):
        """Training class counts of the leaf each sample reaches.

        Returns
        -------
        ndarray of shape (n_samples, n_tree_classes)
        """
        leaves = self.apply(X)        # checks the tree exists first
        return self.tree_.value[leaves]

    def predict(self, X):
        """Predict the class of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
            Labels from ``classes_``. A tie inside a leaf goes to the class
            listed first in ``classes_``.

        Raises
        ------
        NotFittedError
            If no tree has been trained yet.
        """
        counts = self._leaf_counts(X)
        return self._tree_classes[np.argmax(counts, axis=1)]

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_depth(self):
        """Depth of the current tree (0 if it is a single leaf)."""
        self._check_fitted()
        return int(self.tree_.depth.max())

    def get_n_leaves(self):
        """Number of leaves in the current tree."""
        self._check_fitted()
        return int(np.sum(self.tree_.feature < 0))

    def export_text(self, feature_names=None, decimals=3):
        """Describe the current tree as indented text.

        Parameters
        ----------
        feature_names : sequence of str, optional
            Names for the features. Defaults to ``x0, x1, ...``.
        decimals : int, default=3
            Decimal places for thresholds.

        Returns
        -------
        str
        """
        self._check_fitted()
        tree = self.tree_
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.n_features_in_)]
        lines = []

        def describe(node, indent):
            pad = "|   " * indent
            if tree.feature[node] < 0:
                label = self._tree_classes[np.argmax(tree.value[node])].item()
                lines.append(f"{pad}predict {label!r} "
                             f"(samples={tree.n_samples[node]})")
                return
            name = feature_names[tree.feature[node]]
            t = round(float(tree.threshold[node]), decimals)
            lines.append(f"{pad}if {name} <= {t}:")
            describe(tree.left[node], indent + 1)
            lines.append(f"{pad}else:  # {name} > {t}")
            describe(tree.right[node], indent + 1)

        describe(0, 0)
        return "\n".join(lines)
