"""Tests for numcompute_stream.pipeline and numcompute_stream.stream."""

import numpy as np
import pytest

from numcompute_stream.io import chunk_array
from numcompute_stream.metrics import ROCAUC, Accuracy
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier


def binary_stream(n=1200, seed=0, nan_rate=0.05):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    X[rng.random(X.shape) < nan_rate] = np.nan
    return X, y


def make_pipe(buffer_size=300):
    return Pipeline([
        ("impute", Imputer()),
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=4, buffer_size=buffer_size)),
    ])


# ---------------------------------------------------------------- pipeline

def test_pipeline_partial_fit_updates_every_step_and_predicts():
    X, y = binary_stream()
    pipe = make_pipe()
    for Xc, yc in chunk_array(X, y, chunk_size=100):
        pipe.partial_fit(Xc, yc)
    assert pipe["scale"].n_samples_seen_ == len(X)
    assert pipe["model"].n_retrains_ == 4
    assert pipe.predict(X).shape == (len(X),)
    assert pipe.predict_proba(X).shape == (len(X), 2)
    assert pipe.score(X, y) > 0.8


def test_pipeline_streaming_matches_manual_chaining():
    X, y = binary_stream(n=300)
    pipe = make_pipe(buffer_size=300)
    imp, sc = Imputer(), StandardScaler()
    tree = DecisionTreeClassifier(max_depth=4, buffer_size=300)
    for Xc, yc in chunk_array(X, y, chunk_size=100):
        pipe.partial_fit(Xc, yc)
        Xt = sc.partial_fit(imp.partial_fit(Xc).transform(Xc)).transform(
            imp.transform(Xc))
        tree.partial_fit(Xt, yc)
    assert np.array_equal(pipe["model"].tree_.feature, tree.tree_.feature)


def test_pipeline_validates_steps():
    with pytest.raises(ValueError, match="non-empty"):
        Pipeline([])
    with pytest.raises(ValueError, match="unique"):
        Pipeline([("s", StandardScaler()), ("s", DecisionTreeClassifier())])
    with pytest.raises(ValueError, match="must have a transform"):
        Pipeline([("m", DecisionTreeClassifier()), ("s", StandardScaler())])
    with pytest.raises(ValueError, match="no partial_fit"):
        Pipeline([("x", object())])


def test_pipeline_clone_and_reset_give_unfitted_copies():
    X, y = binary_stream(n=300)
    pipe = make_pipe(buffer_size=100)
    pipe.partial_fit(X, y)
    copy = pipe.clone()
    assert copy["model"].tree_ is None and copy["model"].buffer_size == 100
    pipe.reset()
    assert pipe["model"].tree_ is None and pipe["scale"].mean_ is None


# ---------------------------------------------------------------- trainer

def test_trainer_scores_before_training_test_then_train():
    X, y = binary_stream()
    trainer = StreamTrainer(make_pipe(buffer_size=300))
    history = trainer.run(chunk_array(X, y, chunk_size=100))
    # Chunks 1-3 fill the first buffer, so there is no model to score with.
    assert [r["scored"] for r in history[:4]] == [False, False, False, True]
    assert np.isnan(history[0]["chunk_accuracy"])
    # The first retrain happens while *fitting* chunk 3.
    assert trainer.retrain_chunks() == [3, 6, 9, 12]


def test_trainer_cumulative_accuracy_counts_only_scored_samples():
    X, y = binary_stream()
    model = make_pipe(buffer_size=300)
    trainer = StreamTrainer(model)
    trainer.run(chunk_array(X, y, chunk_size=100))
    scored = [r for r in trainer.history_ if r["scored"]]
    weights = np.array([r["n_samples"] for r in scored])
    accs = np.array([r["chunk_accuracy"] for r in scored])
    expected = np.sum(weights * accs) / weights.sum()
    assert trainer.history_[-1]["cumulative_accuracy"] == pytest.approx(expected)
    assert trainer.history_[-1]["accuracy"] == pytest.approx(expected)


def test_trainer_logs_memory_timings_and_auc():
    X, y = binary_stream()
    trainer = StreamTrainer(make_pipe(), metrics={
        "accuracy": Accuracy(), "auc": ROCAUC(pos_label=1)})
    trainer.run(chunk_array(X, y, chunk_size=150))
    last = trainer.history_[-1]
    assert last["memory_bytes"] > 0
    assert last["fit_seconds"] >= 0
    assert 0.5 < last["auc"] <= 1.0
    summary = trainer.summary()
    assert summary["retrains"] == trainer.history_[-1]["n_retrains"]
    assert summary["samples_seen"] == len(X)
    assert trainer.get("chunk_accuracy").shape == (len(trainer.history_),)
    with pytest.raises(KeyError, match="not logged"):
        trainer.get("speed")


def test_trainer_verbose_prints_one_line_per_chunk(capsys):
    X, y = binary_stream(n=400)
    StreamTrainer(make_pipe(buffer_size=200), verbose=True).run(
        chunk_array(X, y, chunk_size=100))
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4
    assert "RETRAINED" in lines[1]


def test_trainer_rejects_bad_model_and_metric():
    with pytest.raises(TypeError, match="partial_fit and predict"):
        StreamTrainer(object())
    with pytest.raises(TypeError, match="update and result"):
        StreamTrainer(make_pipe(), metrics={"bad": object()})


def test_larger_buffer_means_fewer_retrains():
    X, y = binary_stream(n=2000)
    counts = []
    for size in (100, 500, 1000):
        tree = DecisionTreeClassifier(max_depth=4, buffer_size=size)
        trainer = StreamTrainer(tree)
        trainer.run(chunk_array(np.nan_to_num(X), y, chunk_size=100))
        counts.append(trainer.summary()["retrains"])
    assert counts == [20, 4, 2]
