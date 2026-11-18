"""Tests for numcompute_stream.visualise (plots run without a screen)."""

import matplotlib

matplotlib.use("Agg")  # draw off-screen so tests run anywhere

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from numcompute_stream.visualise import (  # noqa: E402
    compare_models,
    plot_benchmark,
    plot_confusion_matrix,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_metric_over_time_saves_file_and_keeps_nan_gaps(tmp_path):
    path = tmp_path / "acc.png"
    values = [np.nan, np.nan, 0.7, 0.8, 0.85]
    fig, ax = plot_metric_over_time(values, "Accuracy", "accuracy",
                                    retrain_points=[2, 4], save_path=path)
    assert path.exists() and path.stat().st_size > 0
    assert ax.get_title(loc="left") == "Accuracy"
    line = ax.get_lines()[0]
    assert np.isnan(line.get_ydata()[0])          # gap, not a fake zero
    assert len(ax.get_lines()) == 3               # metric + 2 retrain markers


def test_compare_models_draws_each_series_with_legend():
    fig, ax = compare_models([0.6, 0.7], [0.5, 0.65, 0.7],
                             labels=["buffer 100", "buffer 500"])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "buffer 100", "buffer 500"]
    with pytest.raises(ValueError, match="Expected 3 labels"):
        compare_models([1], [1], labels=["a", "b"], more=[[1]])


def test_plot_predictions_vs_ground_truth_with_text_labels(tmp_path):
    y_true = np.array(["cat", "dog", "dog", "fox"])
    y_pred = np.array(["cat", "dog", "cat", "fox"])
    fig, axes = plot_predictions_vs_ground_truth(
        y_true, y_pred, save_path=tmp_path / "pred.png")
    assert len(axes) == 2
    assert "75.0%" in fig._suptitle.get_text()
    with pytest.raises(ValueError, match="different lengths"):
        plot_predictions_vs_ground_truth([1, 2], [1])


def test_confusion_matrix_and_benchmark_plots_validate_input():
    fig, ax = plot_confusion_matrix([[3, 1], [0, 4]], ["no", "yes"],
                                    normalize=True)
    assert ax.texts[1].get_text() == "0.25"
    with pytest.raises(ValueError, match="square"):
        plot_confusion_matrix([[1, 2, 3]])
    fig, ax = plot_benchmark([100, 1000], {"loop": [0.1, 2.0],
                                           "vectorised": [0.01, 0.05]},
                             "rows", "seconds", log_y=True)
    assert ax.get_yscale() == "log"
    with pytest.raises(ValueError, match="1-D"):
        plot_metric_over_time([[1, 2], [3, 4]])
