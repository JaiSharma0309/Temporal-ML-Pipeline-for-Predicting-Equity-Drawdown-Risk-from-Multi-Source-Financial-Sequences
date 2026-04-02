"""
model_visualizations.py
=======================
Creates three evaluation visuals for stage-1 drawdown-risk models:

1) Top-decile lift by walk-forward fold (bar chart)
2) Fold-by-fold ROC AUC + PR AUC (bar charts)
3) Test-set cumulative lift curve (best model vs baseline)

Inputs expected in `results/stage1/`:
- walk_forward_cv_results.csv
- {model}_test_predictions.csv files for BEST_TEST_MODEL and BASELINE_MODEL

Usage:
    python model_visualizations.py

Outputs:
    results/stage1/plots/
        01_top_decile_lift_by_fold.png
        02_fold_by_fold_roc_pr.png
        03_test_cumulative_lift_curve.png
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

RESULTS_DIR = REPO_ROOT / "results/stage1"
OUT_DIR = RESULTS_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
WALK_FWD_PATH = RESULTS_DIR / "walk_forward_cv_results.csv"

# Models to show in fold-based bar charts
MODELS_FOR_FOLD_PLOTS = [
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "ensemble_rf_lr",
]

# Models used for the cumulative lift curve on test set
BEST_TEST_MODEL = "random_forest"
BASELINE_MODEL = "dummy"


def fold_sort_key(fold_label: str) -> int:
    """
    Extract the first year from a fold label for chronological sorting.

    Supports labels like:
      - val_2020-2020
      - val_2020–2020
    """
    match = re.search(r"(\d{4})", str(fold_label))
    return int(match.group(1)) if match else 9999


def load_walk_forward() -> pd.DataFrame:
    """
    Load walk-forward CV results and keep classifier rows only.

    Returns:
        DataFrame with folds sorted by extracted year.
    """
    df = pd.read_csv(WALK_FWD_PATH)
    df = df[df["model_type"] == "classifier"].copy()
    df["fold_order"] = df["fold"].apply(fold_sort_key)
    df = df.sort_values(["fold_order", "model"]).reset_index(drop=True)
    return df


def plot_top_decile_lift_by_fold(df: pd.DataFrame) -> None:
    """
    Create a bar chart of lift@10% by fold for selected classifier models.

    Args:
        df: Classifier-only walk-forward CV DataFrame.
    """
    d = df[df["model"].isin(MODELS_FOR_FOLD_PLOTS)].copy()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=d, x="fold", y="top10_lift", hue="model")
    plt.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Random baseline (1.0x)",
    )
    plt.title("Top-Decile Lift by Fold (Walk-Forward CV)")
    plt.xlabel("Validation Fold")
    plt.ylabel("Lift@10%")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_top_decile_lift_by_fold.png", dpi=200)
    plt.close()


def plot_fold_by_fold_roc_pr(df: pd.DataFrame) -> None:
    """
    Create side-by-side bar charts for fold ROC AUC and PR AUC.

    Args:
        df: Classifier-only walk-forward CV DataFrame.
    """
    d = df[df["model"].isin(MODELS_FOR_FOLD_PLOTS)].copy()

    dm = d.melt(
        id_vars=["fold", "model"],
        value_vars=["val_roc_auc", "val_pr_auc"],
        var_name="metric",
        value_name="value",
    )
    dm["metric"] = dm["metric"].map(
        {
            "val_roc_auc": "ROC AUC",
            "val_pr_auc": "PR AUC",
        }
    )

    grid = sns.catplot(
        data=dm,
        kind="bar",
        x="fold",
        y="value",
        hue="model",
        col="metric",
        height=5,
        aspect=1.2,
        sharey=False,
    )
    grid.fig.subplots_adjust(top=0.85)
    grid.fig.suptitle("Fold-by-Fold ROC AUC and PR AUC (Walk-Forward CV)")

    for ax in grid.axes.flat:
        ax.set_xlabel("Validation Fold")
        ax.set_ylabel("Score")

    grid.savefig(OUT_DIR / "02_fold_by_fold_roc_pr.png", dpi=200)
    plt.close("all")


def cumulative_lift_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute cumulative gains/lift curve coordinates from labels and scores.

    Args:
        y_true: Binary labels (1 = event, 0 = non-event).
        y_score: Model risk scores, higher = riskier.

    Returns:
        x: Fraction of population screened (0..1)
        y: Fraction of events captured (0..1)
    """
    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    n = len(y_sorted)
    x = np.arange(1, n + 1) / n

    cum_pos = np.cumsum(y_sorted)
    total_pos = max(cum_pos[-1], 1)
    y = cum_pos / total_pos

    return x, y


def load_predictions(model_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load test predictions for one model.

    Expected file format:
        results/stage1/{model_name}_test_predictions.csv
    with columns including `y_true` and `y_score`.
    """
    path = RESULTS_DIR / f"{model_name}_test_predictions.csv"
    df = pd.read_csv(path)
    return df["y_true"].to_numpy(), df["y_score"].to_numpy()


def plot_test_cumulative_lift_curve() -> None:
    """
    Plot test-set cumulative lift curve for best model vs baseline.

    Also marks the top-10% operating point for the best model.
    """
    y_true_best, y_score_best = load_predictions(BEST_TEST_MODEL)
    y_true_base, y_score_base = load_predictions(BASELINE_MODEL)

    x_best, y_best = cumulative_lift_curve(y_true_best, y_score_best)
    x_base, y_base = cumulative_lift_curve(y_true_base, y_score_base)

    plt.figure(figsize=(8, 6))
    plt.plot(x_best, y_best, label=f"{BEST_TEST_MODEL} (test)")
    plt.plot(x_base, y_base, label=f"{BASELINE_MODEL} (test)", alpha=0.9)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Random ranker")

    k = int(np.ceil(0.10 * len(y_true_best)))
    top_idx = np.argsort(-y_score_best)[:k]
    capture_at_10 = np.sum(y_true_best[top_idx]) / max(np.sum(y_true_best), 1)

    plt.scatter([0.10], [capture_at_10], s=50)
    plt.annotate(
        f"Top 10% capture: {capture_at_10:.1%}",
        (0.10, capture_at_10),
        xytext=(0.14, capture_at_10),
    )

    plt.title("Cumulative Lift Curve on Test Set")
    plt.xlabel("Fraction of population screened")
    plt.ylabel("Fraction of drawdowns captured")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_test_cumulative_lift_curve.png", dpi=200)
    plt.close()


def main() -> None:
    """
    Orchestrate all visualization generation.

    Reads walk-forward metrics and prediction files, creates all three plots,
    and writes them to `results/stage1/plots/`.
    """
    sns.set_theme(style="whitegrid", context="talk")

    wf = load_walk_forward()
    plot_top_decile_lift_by_fold(wf)
    plot_fold_by_fold_roc_pr(wf)
    plot_test_cumulative_lift_curve()

    print(f"Saved plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()
