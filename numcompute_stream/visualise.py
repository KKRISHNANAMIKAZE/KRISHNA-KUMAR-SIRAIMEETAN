"""
Reusable matplotlib plots for streaming experiments.

Every function:

* draws on ``ax`` if one is given (so plots can be combined into one
  figure), otherwise creates a new figure;
* saves the figure if ``save_path`` is given (``.png``, ``.pdf``, ...);
* calls ``plt.show()`` if ``show=True`` (in Jupyter the figure also
  appears automatically at the end of the cell);
* returns ``(fig, ax)`` so the caller can change anything afterwards.

Colours come from a fixed, colour-blind-checked order and every series also
has its own marker or line style, so no plot relies on colour alone.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "compare_models",
    "plot_benchmark",
    "plot_confusion_matrix",
    "plot_metric_over_time",
    "plot_predictions_vs_ground_truth",
]

#: Series colours, always used in this order (validated for colour blindness).
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
           "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":"]
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
WRONG = "#d03b3b"
# Light-to-dark blue for magnitudes (confusion matrix).
BLUES = ["#f7fafe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
         "#184f95", "#0d366b"]


def _new_axes(ax, figsize=(8, 4)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=TEXT, loc="left", fontsize=12)
    ax.set_xlabel(xlabel, color=TEXT_MUTED)
    ax.set_ylabel(ylabel, color=TEXT_MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED)


def _finish(fig, save_path, show):
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


def _as_series(values, name):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D (one value per chunk); got "
                         f"shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr


def _mark_retrains(ax, retrain_points):
    for i, chunk in enumerate(retrain_points or []):
        ax.axvline(chunk, color=TEXT_MUTED, linestyle=":", linewidth=1,
                   alpha=0.6, label="retrain" if i == 0 else None)


def plot_metric_over_time(metric_values, title="Metric over time",
                          ylabel="Value", xlabel="Chunk", label=None,
                          retrain_points=None, ax=None, save_path=None,
                          show=False):
    """Plot one metric (for example accuracy) across chunks.

    Parameters
    ----------
    metric_values : array-like of shape (n_chunks,)
        One value per chunk. NaN values (for example chunks before the
        first tree was trained) are left as gaps.
    title, ylabel, xlabel : str
        Text for the plot.
    label : str, optional
        Legend name for the line.
    retrain_points : sequence of int, optional
        Chunk numbers where the model retrained; drawn as dotted lines.
    ax : matplotlib Axes, optional
        Axes to draw on.
    save_path : str or path, optional
        File to save the figure to.
    show : bool, default=False
        Call ``plt.show()``.

    Returns
    -------
    (Figure, Axes)
    """
    values = _as_series(metric_values, "metric_values")
    fig, ax = _new_axes(ax)
    x = np.arange(1, values.size + 1)
    marker = "o" if values.size <= 40 else None    # dots only on short series
    ax.plot(x, values, color=COLOURS[0], linewidth=2, marker=marker,
            markersize=4, label=label)
    _mark_retrains(ax, retrain_points)
    _style(ax, title, xlabel, ylabel)
    if label is not None or retrain_points:
        ax.legend(frameon=False, labelcolor=TEXT)
    _finish(fig, save_path, show)
    return fig, ax


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"),
                   title="Model comparison", ylabel="Accuracy",
                   xlabel="Chunk", more=None, colours=None, ax=None,
                   save_path=None, show=False):
    """Compare two (or more) models on a streaming metric.

    Parameters
    ----------
    metric1, metric2 : array-like of shape (n_chunks,)
        Metric values per chunk for each model (lengths may differ).
    labels : sequence of str
        Legend names, one per series (including any in ``more``).
    title, ylabel, xlabel : str
    more : list of array-like, optional
        Extra series to compare, for example a third ``buffer_size``.
    colours : list of str, optional
        One colour per series. Pass these when the same models appear in
        several plots, so each model keeps one colour everywhere.
    ax, save_path, show
        See :func:`plot_metric_over_time`.

    Returns
    -------
    (Figure, Axes)
    """
    series = [_as_series(metric1, "metric1"), _as_series(metric2, "metric2")]
    series += [_as_series(m, "more") for m in (more or [])]
    if len(labels) != len(series):
        raise ValueError(f"Expected {len(series)} labels; got {len(labels)}")
    if len(series) > len(COLOURS):
        raise ValueError(f"At most {len(COLOURS)} series can be compared")
    colours = COLOURS[:len(series)] if colours is None else list(colours)
    if len(colours) != len(series):
        raise ValueError(f"Expected {len(series)} colours; got {len(colours)}")
    fig, ax = _new_axes(ax)
    for i, (values, name) in enumerate(zip(series, labels)):
        ax.plot(np.arange(1, values.size + 1), values, color=colours[i],
                linewidth=2, linestyle=LINESTYLES[i % len(LINESTYLES)],
                marker=MARKERS[i], markersize=4, markevery=max(1, values.size // 15),
                label=name)
    _style(ax, title, xlabel, ylabel)
    ax.legend(frameon=False, labelcolor=TEXT)
    _finish(fig, save_path, show)
    return fig, ax


def plot_confusion_matrix(cm, labels=None, normalize=False,
                          title="Confusion matrix", ax=None, save_path=None,
                          show=False):
    """Draw a confusion matrix as a heatmap with the count in each cell.

    Parameters
    ----------
    cm : array-like of shape (n_classes, n_classes)
        Rows are true classes, columns are predicted classes.
    labels : sequence, optional
        Class names for the rows and columns.
    normalize : bool, default=False
        Show each row as fractions of that true class.
    title, ax, save_path, show
        See :func:`plot_metric_over_time`.

    Returns
    -------
    (Figure, Axes)
    """
    cm = np.asarray(cm, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm must be a square 2-D array; got shape {cm.shape}")
    k = cm.shape[0]
    labels = [str(v) for v in (range(k) if labels is None else labels)]
    if len(labels) != k:
        raise ValueError(f"Expected {k} labels; got {len(labels)}")
    shown = cm
    if normalize:
        totals = cm.sum(axis=1, keepdims=True)
        shown = np.divide(cm, totals, out=np.zeros_like(cm), where=totals > 0)

    from matplotlib.colors import ListedColormap
    fig, ax = _new_axes(ax, figsize=(4.5, 4))
    ax.imshow(shown, cmap=ListedColormap(BLUES), vmin=0,
              vmax=max(shown.max(), 1e-12))
    top = shown.max() if shown.max() > 0 else 1.0
    for i in range(k):
        for j in range(k):
            text = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            dark = shown[i, j] / top > 0.55
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if dark else TEXT, fontsize=10)
    ax.set_xticks(range(k), labels)
    ax.set_yticks(range(k), labels)
    ax.set_xlabel("Predicted", color=TEXT_MUTED)
    ax.set_ylabel("True", color=TEXT_MUTED)
    ax.set_title(title, color=TEXT, loc="left", fontsize=12)
    ax.tick_params(colors=TEXT_MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _finish(fig, save_path, show)
    return fig, ax


def plot_predictions_vs_ground_truth(y_true, y_pred, title=None,
                                     max_points=200, save_path=None,
                                     show=False):
    """Show predictions against the true labels for one chunk.

    The left panel plots every sample (up to ``max_points``): a hollow
    circle at its true class and a cross at its predicted class; wrong
    predictions are drawn in red with a larger cross. The right panel is
    the chunk's confusion matrix.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        True and predicted labels (numbers or text).
    title : str, optional
        Overall title. Defaults to the chunk accuracy.
    max_points : int, default=200
        Only the first ``max_points`` samples are drawn in the left panel.
    save_path, show
        See :func:`plot_metric_over_time`.

    Returns
    -------
    (Figure, array of two Axes)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred have different lengths: "
                         f"{y_true.size} and {y_pred.size}")
    if y_true.size == 0:
        raise ValueError("y_true is empty")
    labels = np.union1d(y_true, y_pred)
    t = np.searchsorted(labels, y_true)
    p = np.searchsorted(labels, y_pred)
    wrong = t != p

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    n = min(y_true.size, max_points)
    idx = np.arange(n)
    ax_left.scatter(idx, t[:n], facecolors="none", edgecolors=COLOURS[0],
                    s=40, linewidths=1.5, label="true class")
    ok = ~wrong[:n]
    ax_left.scatter(idx[ok], p[:n][ok], marker="x", color=COLOURS[0], s=20,
                    linewidths=1.2, label="predicted (correct)")
    ax_left.scatter(idx[~ok], p[:n][~ok], marker="x", color=WRONG, s=60,
                    linewidths=2, label="predicted (wrong)")
    ax_left.set_yticks(range(len(labels)), [str(v) for v in labels])
    _style(ax_left, f"First {n} samples of the chunk", "Sample", "Class")
    # The band between class rows is empty, so the legend goes there.
    ax_left.legend(frameon=True, facecolor="white", edgecolor=GRID,
                   labelcolor=TEXT, loc="center", fontsize=8, ncol=3)

    k = len(labels)
    cm = np.bincount(t * k + p, minlength=k * k).reshape(k, k)
    plot_confusion_matrix(cm, labels, title="Confusion matrix", ax=ax_right)

    accuracy = 1.0 - wrong.mean()
    fig.suptitle(title or f"Predictions vs ground truth (accuracy {accuracy:.1%})",
                 color=TEXT, x=0.01, ha="left", fontsize=13)
    _finish(fig, save_path, show)
    return fig, np.array([ax_left, ax_right])


def plot_benchmark(x, series, xlabel, ylabel, title="Benchmark", log_y=False,
                   ax=None, save_path=None, show=False):
    """Line chart for benchmark results (one line per implementation).

    Parameters
    ----------
    x : array-like
        Values on the x axis (for example input sizes).
    series : dict of str to array-like
        Name to y values (same length as ``x``).
    xlabel, ylabel, title : str
    log_y : bool, default=False
        Use a log scale on the y axis (useful for timings).
    ax, save_path, show
        See :func:`plot_metric_over_time`.

    Returns
    -------
    (Figure, Axes)
    """
    x = np.asarray(x)
    if len(series) > len(COLOURS):
        raise ValueError(f"At most {len(COLOURS)} series can be drawn")
    fig, ax = _new_axes(ax)
    for i, (name, values) in enumerate(series.items()):
        values = np.asarray(values, dtype=float)
        if values.shape != x.shape:
            raise ValueError(f"Series {name!r} has {values.size} values; "
                             f"x has {x.size}")
        ax.plot(x, values, color=COLOURS[i], linewidth=2, marker=MARKERS[i],
                markersize=6, linestyle=LINESTYLES[i % len(LINESTYLES)],
                label=name)
    if log_y:
        ax.set_yscale("log")
    _style(ax, title, xlabel, ylabel)
    ax.legend(frameon=False, labelcolor=TEXT)
    _finish(fig, save_path, show)
    return fig, ax
