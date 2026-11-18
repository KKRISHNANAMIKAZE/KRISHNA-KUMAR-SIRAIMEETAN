"""
Benchmark (b): how buffer_size changes accuracy and retraining cost.

The Electricity data is streamed in chunks of 250 rows (about 5 days each)
through the same pipeline (Imputer -> StandardScaler -> decision tree,
max_depth=5), changing only ``buffer_size``. Evaluation is test-then-train:
each chunk is predicted before the model learns from it.

For each buffer size it records the final cumulative accuracy, the mean
rolling accuracy, the number of retrains, total and average retraining
time, and peak model memory.

Run from the project folder:

    python benchmark/bench_buffer_size.py

Results go to benchmark/results/ (CSV + PNG).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import DATA, RESULTS, environment, write_csv  # noqa: E402
from numcompute_stream import (  # noqa: E402
    DecisionTreeClassifier,
    Imputer,
    Pipeline,
    StandardScaler,
    StreamTrainer,
    iter_csv_chunks,
)
from numcompute_stream.visualise import (  # noqa: E402
    COLOURS,
    compare_models,
    plot_benchmark,
)

CHUNK_SIZE = 250
BUFFER_SIZES = [250, 500, 1000, 2500, 5000]


def run_one(buffer_size):
    pipe = Pipeline([
        ("impute", Imputer()),
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=5, buffer_size=buffer_size)),
    ])
    trainer = StreamTrainer(pipe, rolling_window=10)
    trainer.run(iter_csv_chunks(DATA, CHUNK_SIZE, label_column="class"))
    tree = pipe["model"]
    retrain_seconds = [r["seconds"] for r in tree.retrain_log_]
    return trainer, {
        "buffer_size": buffer_size,
        "final_accuracy": trainer.summary()["final_cumulative_accuracy"],
        "mean_rolling_accuracy": float(np.nanmean(trainer.get("rolling_accuracy"))),
        "retrains": tree.n_retrains_,
        "total_retrain_seconds": float(np.sum(retrain_seconds)),
        "mean_retrain_seconds": float(np.mean(retrain_seconds)),
        "first_scored_chunk": int(np.argmax(trainer.get("scored"))) + 1,
        "peak_memory_kb": trainer.summary()["peak_memory_bytes"] / 1024,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("Environment:", environment())
    print(f"Chunk size {CHUNK_SIZE}; buffer sizes {BUFFER_SIZES}\n")
    trainers, results = [], []
    for size in BUFFER_SIZES:
        trainer, res = run_one(size)
        trainers.append(trainer)
        results.append(res)
        print(f"buffer {size:>5}: final acc {res['final_accuracy']:.4f} | "
              f"mean rolling acc {res['mean_rolling_accuracy']:.4f} | "
              f"retrains {res['retrains']:>3} | total retrain "
              f"{res['total_retrain_seconds']:.3f}s | per retrain "
              f"{res['mean_retrain_seconds'] * 1000:.1f} ms | peak mem "
              f"{res['peak_memory_kb']:.0f} KB")

    header = list(results[0])
    write_csv(RESULTS / "buffer_size.csv", header,
              [[r[h] for h in header] for r in results])

    labels = [f"buffer {s}" for s in BUFFER_SIZES]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    rolling = [t.get("rolling_accuracy") for t in trainers]
    compare_models(rolling[0], rolling[2], labels=[labels[0], labels[2], labels[4]],
                   more=[rolling[4]], colours=[COLOURS[0], COLOURS[2], COLOURS[4]],
                   title="Rolling accuracy (last 10 chunks)",
                   ylabel="accuracy", ax=axes[0, 0])
    compare_models(*[t.get("cumulative_accuracy") for t in trainers[:2]],
                   labels=labels, more=[t.get("cumulative_accuracy")
                                        for t in trainers[2:]],
                   title="Cumulative accuracy", ylabel="accuracy", ax=axes[0, 1])
    plot_benchmark(BUFFER_SIZES,
                   {"total retrain time (s)": [r["total_retrain_seconds"]
                                               for r in results],
                    "time per retrain (s)": [r["mean_retrain_seconds"]
                                             for r in results]},
                   "buffer_size", "seconds (log scale)", "Retraining cost",
                   log_y=True, ax=axes[1, 0])
    plot_benchmark(BUFFER_SIZES,
                   {"final cumulative accuracy": [r["final_accuracy"]
                                                  for r in results]},
                   "buffer_size", "accuracy", "Accuracy vs buffer size",
                   ax=axes[1, 1])
    fig.suptitle("Effect of buffer_size on the Electricity stream", x=0.01,
                 ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS / "buffer_size.png", dpi=150)
    print(f"\nSaved results to {RESULTS}")


if __name__ == "__main__":
    main()
