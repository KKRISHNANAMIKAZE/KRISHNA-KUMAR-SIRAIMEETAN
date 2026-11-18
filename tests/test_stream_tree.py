"""Tests for numcompute_stream.tree (buffered decision tree)."""

import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier, impurity
from numcompute_stream.utils import NotFittedError

# The 12 students from the course guide: submitted, attended, study_group,
# passed. The rule is "passed if submitted AND attended", except student 5.
STUDENTS = np.array([
    [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1], [1, 1, 0, 1],
    [1, 0, 1, 1],
    [1, 0, 0, 0], [1, 0, 0, 0],
    [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0],
])
X_STUDENTS, Y_STUDENTS = STUDENTS[:, :3], STUDENTS[:, 3]


def make_blobs(n=300, seed=0, n_features=4, n_classes=3):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    X = rng.normal(size=(n, n_features)) + y[:, None] * 1.5
    return X, y


# ---------------------------------------------------------------- algorithm

def test_impurity_known_values():
    assert impurity(np.array([6.0, 6.0]), 12.0, "gini") == pytest.approx(0.5)
    assert impurity(np.array([6.0, 6.0]), 12.0, "entropy") == pytest.approx(1.0)
    assert impurity(np.array([5.0, 0.0]), 5.0, "gini") == 0.0
    assert impurity(np.array([5.0, 0.0]), 5.0, "entropy") == 0.0


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_course_example_depth_two_matches_guide(criterion):
    tree = DecisionTreeClassifier(max_depth=2, criterion=criterion)
    tree.fit(X_STUDENTS, Y_STUDENTS)
    t = tree.tree_
    assert t.feature[0] == 0                          # root asks "submitted?"
    did_submit = t.right[0]                           # submitted > 0.5
    assert t.feature[did_submit] == 1                 # then "attended?"
    assert t.feature[t.left[0]] == -1                 # did not submit: leaf
    # The depth-2 tree is exactly the rule "submitted AND attended".
    grid = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
    assert np.array_equal(tree.predict(grid), grid[:, 0] & grid[:, 1])


def test_course_example_deep_tree_learns_the_exception():
    tree = DecisionTreeClassifier(max_depth=3).fit(X_STUDENTS, Y_STUDENTS)
    assert tree.score(X_STUDENTS, Y_STUDENTS) == 1.0   # student 5 learned
    shallow = DecisionTreeClassifier(max_depth=2).fit(X_STUDENTS, Y_STUDENTS)
    assert shallow.score(X_STUDENTS, Y_STUDENTS) == pytest.approx(11 / 12)


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_loop_and_vectorized_build_identical_trees(criterion, seed):
    rng = np.random.default_rng(seed)
    X = np.round(rng.normal(size=(150, 5)), 1)       # rounding creates ties
    y = rng.integers(0, 4, 150)
    fast = DecisionTreeClassifier(max_depth=None, criterion=criterion,
                                  implementation="vectorized").fit(X, y)
    slow = DecisionTreeClassifier(max_depth=None, criterion=criterion,
                                  implementation="loop").fit(X, y)
    for field in fast.tree_._fields:
        assert np.array_equal(getattr(fast.tree_, field),
                              getattr(slow.tree_, field), equal_nan=True)
    assert np.array_equal(fast.apply(X), slow.apply(X))


def test_threshold_is_midpoint_between_neighbouring_values():
    X = np.array([[1.0], [2.0], [10.0], [11.0]])
    tree = DecisionTreeClassifier().fit(X, [0, 0, 1, 1])
    assert tree.tree_.threshold[0] == 6.0
    assert tree.predict([[5.9], [6.1]]).tolist() == [0, 1]


def test_gain_tie_goes_to_lowest_feature_index():
    col = np.array([0.0, 0.0, 1.0, 1.0])
    X = np.column_stack([col, col])                   # two identical features
    tree = DecisionTreeClassifier().fit(X, [0, 0, 1, 1])
    assert tree.tree_.feature[0] == 0


def test_leaf_tie_goes_to_first_class():
    X = np.zeros((4, 1))                              # no split possible
    assert DecisionTreeClassifier().fit(X, [1, 0, 1, 0]).predict(X[:1])[0] == 0
    tree = DecisionTreeClassifier().fit(X, ["b", "a", "b", "a"])
    assert tree.predict(X[:1])[0] == "a"
    assert tree.get_n_leaves() == 1


def test_max_depth_and_min_samples_split_are_respected():
    X, y = make_blobs()
    assert DecisionTreeClassifier(max_depth=2).fit(X, y).get_depth() <= 2
    tree = DecisionTreeClassifier(max_depth=None, min_samples_split=50).fit(X, y)
    internal = tree.tree_.feature >= 0
    assert np.all(tree.tree_.n_samples[internal] >= 50)


def test_string_labels_and_probabilities():
    X, y = make_blobs()
    names = np.array(["cat", "dog", "fox"])[y]
    tree = DecisionTreeClassifier(max_depth=4).fit(X, names)
    assert set(tree.predict(X)) <= {"cat", "dog", "fox"}
    proba = tree.predict_proba(X)
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(tree.classes_[proba.argmax(axis=1)], tree.predict(X))


def test_max_features_subset_is_repeatable_with_random_state():
    X, y = make_blobs(n_features=8)
    a = DecisionTreeClassifier(max_features="sqrt", random_state=3).fit(X, y)
    b = DecisionTreeClassifier(max_features="sqrt", random_state=3).fit(X, y)
    assert np.array_equal(a.tree_.feature, b.tree_.feature)


# ---------------------------------------------------------------- buffering

def test_no_tree_until_buffer_is_full_and_clear_error_before():
    X, y = make_blobs(n=100)
    tree = DecisionTreeClassifier(buffer_size=100)
    tree.partial_fit(X[:60], y[:60])
    assert tree.n_retrains_ == 0
    assert tree.buffer_count_ == 60
    with pytest.raises(NotFittedError, match="60 of 100 samples"):
        tree.predict(X[:5])


def test_full_buffer_triggers_retrain_and_clears_buffer():
    X, y = make_blobs(n=100)
    tree = DecisionTreeClassifier(buffer_size=100)
    tree.partial_fit(X[:60], y[:60]).partial_fit(X[60:], y[60:])
    assert tree.n_retrains_ == 1
    assert tree.buffer_count_ == 0
    assert tree.retrain_log_[0]["n_samples"] == 100


def test_predict_uses_old_tree_between_retrains():
    X, y = make_blobs(n=200)
    tree = DecisionTreeClassifier(buffer_size=100)
    tree.partial_fit(X[:100], y[:100])
    before = tree.predict(X)
    tree.partial_fit(X[100:150], (y[100:150] + 1) % 3)   # buffered only
    assert np.array_equal(tree.predict(X), before)
    assert tree.n_retrains_ == 1


def test_chunk_bigger_than_buffer_retrains_once_on_all_of_it():
    X, y = make_blobs(n=250)
    tree = DecisionTreeClassifier(buffer_size=100).partial_fit(X, y)
    assert tree.n_retrains_ == 1
    assert tree.retrain_log_[0]["n_samples"] == 250
    assert tree.buffer_count_ == 0


def test_partial_fit_on_one_full_buffer_equals_fit():
    X, y = make_blobs(n=120)
    streamed = DecisionTreeClassifier(buffer_size=120)
    for start in range(0, 120, 40):
        streamed.partial_fit(X[start:start + 40], y[start:start + 40])
    batch = DecisionTreeClassifier().fit(X, y)
    for field in batch.tree_._fields:
        assert np.array_equal(getattr(streamed.tree_, field),
                              getattr(batch.tree_, field), equal_nan=True)


def test_retrain_uses_only_buffered_data():
    X, y = make_blobs(n=200)
    tree = DecisionTreeClassifier(buffer_size=100)
    tree.partial_fit(X[:100], y[:100]).partial_fit(X[100:], y[100:])
    assert tree.n_retrains_ == 2
    assert tree.tree_.n_samples[0] == 100           # second tree saw 100 rows


def test_flush_trains_on_partial_buffer_and_empty_chunk_is_ignored():
    X, y = make_blobs(n=30)
    tree = DecisionTreeClassifier(buffer_size=500)
    tree.partial_fit(X, y).partial_fit(np.zeros((0, 4)), [])
    assert tree.buffer_count_ == 30
    tree.flush()
    assert tree.n_retrains_ == 1
    assert tree.predict(X).shape == (30,)


def test_single_class_buffer_gives_single_leaf():
    tree = DecisionTreeClassifier(buffer_size=5)
    tree.partial_fit(np.arange(10.0).reshape(5, 2), [1, 1, 1, 1, 1])
    assert tree.get_n_leaves() == 1
    assert tree.predict([[0.0, 0.0]])[0] == 1


def test_new_class_gets_zero_probability_until_next_retrain():
    X, y = make_blobs(n=200, n_classes=2)
    tree = DecisionTreeClassifier(buffer_size=100)
    tree.partial_fit(X[:100], y[:100])
    tree.partial_fit(X[100:110], np.full(10, 7))       # new class 7, buffered
    assert tree.classes_.tolist() == [0, 1, 7]
    proba = tree.predict_proba(X[:5])
    assert proba.shape == (5, 3)
    assert np.all(proba[:, 2] == 0)
    assert np.allclose(proba.sum(axis=1), 1.0)


# ---------------------------------------------------------------- input errors

def test_nan_wrong_width_and_label_type_change_are_rejected():
    tree = DecisionTreeClassifier(buffer_size=10)
    with pytest.raises(ValueError, match="Imputer"):
        tree.partial_fit([[np.nan, 1.0]], [0])
    tree.partial_fit([[1.0, 2.0]], [0])
    with pytest.raises(ValueError, match="X has 3 features"):
        tree.partial_fit([[1.0, 2.0, 3.0]], [0])
    with pytest.raises(TypeError, match="Labels changed type"):
        tree.partial_fit([[1.0, 2.0]], ["a"])


def test_allow_new_features_pads_buffer_and_accepts_wider_input():
    tree = DecisionTreeClassifier(buffer_size=4, allow_new_features=True)
    tree.partial_fit([[0.0], [1.0]], [0, 1])
    tree.partial_fit([[0.0, 1.0], [1.0, 0.0]], [0, 1])   # one new column
    assert tree.n_features_in_ == 2
    assert tree.predict([[0.0, 0.0, 9.0]]).shape == (1,)  # extra col ignored


@pytest.mark.parametrize("bad, error", [
    ({"buffer_size": 0}, ValueError),
    ({"max_depth": 0}, ValueError),
    ({"min_samples_split": 1}, ValueError),
    ({"criterion": "mse"}, ValueError),
    ({"implementation": "fast"}, ValueError),
    ({"max_features": 1.5}, ValueError),
    ({"max_features": "half"}, TypeError),
])
def test_invalid_settings_raise_clear_errors(bad, error):
    with pytest.raises(error):
        DecisionTreeClassifier(**bad)


def test_export_text_reads_like_the_course_tree():
    tree = DecisionTreeClassifier(max_depth=2).fit(X_STUDENTS, Y_STUDENTS)
    text = tree.export_text(["submitted", "attended", "study_group"])
    assert text.splitlines()[0] == "if submitted <= 0.5:"
    assert "attended" in text
