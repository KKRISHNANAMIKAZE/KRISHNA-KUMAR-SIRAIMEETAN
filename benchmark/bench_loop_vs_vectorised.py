"""
Benchmark (a): loop-based versus vectorised implementations.

Four parts of the framework have both a plain Python loop version and a
NumPy vectorised version. For each, this script checks that both give the
same answer, then times them on growing inputs from the Electricity data:

1. Decision tree training (best-split search), max_depth=5.
2. Decision tree prediction (routing samples to leaves).
3. Streaming mean and variance (one-row-at-a-time Welford vs chunk merge).
4. Confusion matrix counting (loop over pairs vs one np.bincount).

Run from the project folder:

    python benchmark/bench_loop_vs_vectorised.py

Results go to benchmark/results/ (CSV + PNG).
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from _common import DATA, RESULTS, environment, median_time, write_csv  # noqa: E402
from numcompute_stream import DecisionTreeClassifier, read_csv  # noqa: E402
from numcompute_stream.metrics import confusion_matrix  # noqa: E402
from numcompute_stream.stats import StreamingStats  # noqa: E402
from numcompute_stream.visualise import plot_benchmark  # noqa: E402


def welford_loop(X):
    """Textbook Welford: update mean and M2 one row at a time."""
    n, mean, m2 = 0, np.zeros(X.shape[1]), np.zeros(X.shape[1])
    for row in X:
        n += 1
        delta = row - mean
        mean += delta / n
        m2 += delta * (row - mean)
    return mean, m2 / n


def welford_vectorised(X, chunk_size=500):
    stats = StreamingStats(quantile_method=None)
    for start in range(0, len(X), chunk_size):
        stats.update_stats(X[start:start + chunk_size])
    return stats.mean_, stats.var_


def confusion_loop(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    X_all, y_all, _ = read_csv(DATA, label_column="class")
    print("Environment:", environment())
    rows = []

    # 1. Tree training ------------------------------------------------
    print("\n1. Tree training (max_depth=5)")
    train_sizes = [250, 500, 1000, 2000, 4000]
    fit_loop, fit_vec = [], []
    for n in train_sizes:
        X, y = X_all[:n], y_all[:n]
        loop = DecisionTreeClassifier(max_depth=5, implementation="loop")
        vec = DecisionTreeClassifier(max_depth=5, implementation="vectorized")
        same = np.array_equal(loop.fit(X, y).tree_.feature,
                              vec.fit(X, y).tree_.feature)
        t_loop = median_time(lambda: loop.fit(X, y), repeats=3)
        t_vec = median_time(lambda: vec.fit(X, y), repeats=5)
        fit_loop.append(t_loop)
        fit_vec.append(t_vec)
        rows.append(["tree_fit", n, t_loop, t_vec, t_loop / t_vec, same])
        print(f"  n={n:>5}: loop {t_loop:8.4f}s  vectorised {t_vec:8.4f}s  "
              f"speed-up {t_loop / t_vec:6.1f}x  identical tree: {same}")

    # 2. Tree prediction ----------------------------------------------
    print("\n2. Tree prediction")
    tree = DecisionTreeClassifier(max_depth=5).fit(X_all[:5000], y_all[:5000])
    predict_sizes = [1000, 5000, 20000, 45312]
    pred_loop, pred_vec = [], []
    for n in predict_sizes:
        X = X_all[:n]
        same = np.array_equal(tree._apply_loop(X), tree._apply_vectorized(X))
        t_loop = median_time(lambda: tree._apply_loop(X), repeats=3)
        t_vec = median_time(lambda: tree._apply_vectorized(X), repeats=5)
        pred_loop.append(t_loop)
        pred_vec.append(t_vec)
        rows.append(["tree_predict", n, t_loop, t_vec, t_loop / t_vec, same])
        print(f"  n={n:>5}: loop {t_loop:8.4f}s  vectorised {t_vec:8.4f}s  "
              f"speed-up {t_loop / t_vec:6.1f}x  identical leaves: {same}")

    # 3. Streaming statistics -----------------------------------------
    print("\n3. Streaming mean and variance")
    stat_sizes = [1000, 5000, 20000, 45312]
    stat_loop, stat_vec = [], []
    for n in stat_sizes:
        X = X_all[:n]
        a, b = welford_loop(X), welford_vectorised(X)
        same = np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
        t_loop = median_time(lambda: welford_loop(X), repeats=3)
        t_vec = median_time(lambda: welford_vectorised(X), repeats=5)
        stat_loop.append(t_loop)
        stat_vec.append(t_vec)
        rows.append(["stats", n, t_loop, t_vec, t_loop / t_vec, same])
        print(f"  n={n:>5}: loop {t_loop:8.4f}s  vectorised {t_vec:8.4f}s  "
              f"speed-up {t_loop / t_vec:6.1f}x  same result: {same}")

    # 4. Confusion matrix ---------------------------------------------
    print("\n4. Confusion matrix")
    y_pred_all = tree.predict(X_all)
    cm_loop, cm_vec = [], []
    for n in stat_sizes:
        t, p = y_all[:n], y_pred_all[:n]
        same = np.array_equal(confusion_loop(t, p, 2), confusion_matrix(t, p))
        t_loop = median_time(lambda: confusion_loop(t, p, 2), repeats=3)
        t_vec = median_time(lambda: confusion_matrix(t, p), repeats=5)
        cm_loop.append(t_loop)
        cm_vec.append(t_vec)
        rows.append(["confusion_matrix", n, t_loop, t_vec, t_loop / t_vec, same])
        print(f"  n={n:>5}: loop {t_loop:8.4f}s  vectorised {t_vec:8.4f}s  "
              f"speed-up {t_loop / t_vec:6.1f}x  same result: {same}")

    write_csv(RESULTS / "loop_vs_vectorised.csv",
              ["task", "n_samples", "loop_seconds", "vectorised_seconds",
               "speedup", "same_result"], rows)
    (RESULTS / "environment.txt").write_text(environment() + "\n",
                                             encoding="utf-8")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    plot_benchmark(train_sizes, {"loop": fit_loop, "vectorised": fit_vec},
                   "training rows", "seconds (log scale)",
                   "1. Tree training", log_y=True, ax=axes[0, 0])
    plot_benchmark(predict_sizes, {"loop": pred_loop, "vectorised": pred_vec},
                   "rows predicted", "seconds (log scale)",
                   "2. Tree prediction", log_y=True, ax=axes[0, 1])
    plot_benchmark(stat_sizes, {"loop": stat_loop, "vectorised": stat_vec},
                   "rows", "seconds (log scale)",
                   "3. Streaming mean and variance", log_y=True, ax=axes[1, 0])
    plot_benchmark(stat_sizes, {"loop": cm_loop, "vectorised": cm_vec},
                   "rows", "seconds (log scale)",
                   "4. Confusion matrix", log_y=True, ax=axes[1, 1])
    fig.suptitle("Loop vs vectorised (median of repeated runs)", x=0.01,
                 ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS / "loop_vs_vectorised.png", dpi=150)
    print(f"\nSaved results to {RESULTS}")


if __name__ == "__main__":
    main()
