"""
model_visualizations.py
=======================
Creates three evaluation visuals for stage-1 drawdown-risk models:

1) Top-decile lift by walk-forward fold (bar chart)
2) Fold-by-fold ROC AUC + PR AUC (bar charts)
3) Test-set cumulative lift curve (best model vs baseline)
4) Test-set decile event-rate chart (risk bucket view)
5) Test-set capture-vs-workload chart (operations view)
6) Business-impact markdown summary

Inputs expected in `results/stage1/`:
- tables/walk_forward_cv_results.csv
- tables/{model}_test_predictions.csv files for BEST_TEST_MODEL and BASELINE_MODEL

Usage:
    python model_visualizations.py

Outputs:
    results/stage1/plots/
        01_top_decile_lift_by_fold.png
        02_fold_by_fold_roc_pr.png
        03_test_cumulative_lift_curve.png
        04_test_decile_event_rate.png
        05_test_capture_vs_workload.png
    results/stage1/reports/business_impact_summary.md
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
TABLES_DIR = RESULTS_DIR / "tables"
OUT_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
WALK_FWD_PATH = TABLES_DIR / "walk_forward_cv_results.csv"
BEST_MODEL_PATH = RESULTS_DIR / "best_model.txt"

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
        results/stage1/tables/{model_name}_test_predictions.csv
    with columns including `y_true` and `y_score`.
    """
    path = TABLES_DIR / f"{model_name}_test_predictions.csv"
    df = pd.read_csv(path)
    return df["y_true"].to_numpy(), df["y_score"].to_numpy()


def load_prediction_df(model_name: str, split: str = "test") -> pd.DataFrame:
    """
    Load per-row prediction table for a model and split.

    Expected file:
        results/stage1/tables/{model_name}_{split}_predictions.csv
    """
    path = TABLES_DIR / f"{model_name}_{split}_predictions.csv"
    return pd.read_csv(path)


def get_best_classifier_name(default: str = BEST_TEST_MODEL) -> str:
    """
    Resolve the best classifier name from disk, falling back to default.
    """
    if BEST_MODEL_PATH.exists():
        name = BEST_MODEL_PATH.read_text().strip()
        if name:
            return name
    return default


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


def build_decile_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a top-to-bottom risk decile table from test predictions.

    Returns one row per decile with count, event count, event rate, and lift.
    Decile 1 = highest risk scores.
    """
    out = df.copy().sort_values("y_score", ascending=False).reset_index(drop=True)
    n = len(out)
    out["rank_frac"] = (np.arange(n) + 1) / n
    out["decile"] = np.ceil(out["rank_frac"] * 10).astype(int)
    out["decile"] = out["decile"].clip(1, 10)

    base_rate = out["y_true"].mean()
    dec = (
        out.groupby("decile", as_index=False)
        .agg(
            n=("y_true", "size"),
            event_count=("y_true", "sum"),
            event_rate=("y_true", "mean"),
        )
        .sort_values("decile")
    )
    dec["lift"] = dec["event_rate"] / base_rate if base_rate > 0 else np.nan
    return dec


def plot_test_decile_event_rate(df_test: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Plot event rate by score decile (1=highest risk), with base-rate baseline.
    """
    dec = build_decile_table(df_test)
    base_rate = float(df_test["y_true"].mean())

    plt.figure(figsize=(10, 6))
    sns.barplot(data=dec, x="decile", y="event_rate", color="#2f6c8f")
    plt.axhline(
        base_rate,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Base rate ({base_rate:.2%})",
    )
    plt.title(f"Test Event Rate by Risk Decile ({model_name})")
    plt.xlabel("Risk Decile (1 = highest predicted risk)")
    plt.ylabel("Observed drawdown rate")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_test_decile_event_rate.png", dpi=200)
    plt.close()
    return dec


def plot_test_capture_vs_workload(df_test: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Plot how many true events are captured as review coverage increases.

    X-axis is workload (fraction of names reviewed, top-scored first).
    Y-axis is fraction of all drawdown events captured.
    """
    d = df_test.sort_values("y_score", ascending=False).reset_index(drop=True)
    y = d["y_true"].to_numpy()
    n = len(y)

    coverage_levels = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    rows = []
    total_events = max(int(y.sum()), 1)
    for c in coverage_levels:
        k = max(1, int(np.ceil(n * c)))
        top = y[:k]
        captured = int(top.sum())
        rows.append(
            {
                "coverage": c,
                "review_n": k,
                "captured_events": captured,
                "capture_rate": captured / total_events,
                "precision_top": float(top.mean()),
            }
        )
    cap = pd.DataFrame(rows)

    plt.figure(figsize=(10, 6))
    plt.plot(cap["coverage"], cap["capture_rate"], marker="o", linewidth=2, label="Event capture rate")
    plt.plot(cap["coverage"], cap["precision_top"], marker="o", linewidth=2, label="Precision in reviewed bucket")
    plt.plot([0, 0.5], [0, 0.5], linestyle="--", color="gray", linewidth=1, label="Random capture")
    plt.title(f"Capture vs Workload on Test Set ({model_name})")
    plt.xlabel("Fraction of names reviewed")
    plt.ylabel("Rate")
    plt.xlim(0.0, 0.5)
    plt.ylim(0.0, 1.0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_test_capture_vs_workload.png", dpi=200)
    plt.close()
    return cap


def write_business_impact_summary(
    model_name: str,
    df_test: pd.DataFrame,
    deciles: pd.DataFrame,
    capture_table: pd.DataFrame,
) -> None:
    """
    Write a concise markdown summary of operational model impact.
    """
    y = df_test["y_true"].to_numpy()
    s = df_test["y_score"].to_numpy()
    base_rate = float(np.mean(y))
    n = len(y)
    total_events = int(np.sum(y))

    k10 = max(1, int(np.ceil(0.10 * n)))
    idx10 = np.argsort(-s)[:k10]
    top10_events = int(np.sum(y[idx10]))
    top10_precision = float(np.mean(y[idx10]))
    top10_capture = top10_events / max(total_events, 1)
    top10_lift = top10_precision / base_rate if base_rate > 0 else np.nan

    threshold_text = "Threshold metrics file not found."
    tpath = TABLES_DIR / f"{model_name}_test_selected_threshold_metrics.csv"
    if tpath.exists():
        tm = pd.read_csv(tpath).iloc[0]
        threshold_text = (
            f"- Selected threshold: `{tm['threshold']:.2f}`\n"
            f"- Precision: `{tm['precision']:.2%}`\n"
            f"- Recall: `{tm['recall']:.2%}`\n"
            f"- False positives / true positive: `{(tm['fp'] / max(tm['tp'], 1)):.2f}`"
        )

    top_dec = deciles.iloc[0]
    last_dec = deciles.iloc[-1]

    lines = [
        "# Business Impact Summary",
        "",
        "## Objective",
        "Prioritize the riskiest names so a risk team can review fewer stocks while catching a large share of severe future drawdowns.",
        "",
        "## Test-Set Impact",
        f"- Model: `{model_name}`",
        f"- Base drawdown rate: `{base_rate:.2%}` (`{total_events:,}` events over `{n:,}` rows)",
        f"- Top 10% reviewed: `{k10:,}` rows",
        f"- Precision in top 10%: `{top10_precision:.2%}`",
        f"- Lift at top 10%: `{top10_lift:.2f}x`",
        f"- Share of all drawdowns captured by top 10%: `{top10_capture:.2%}`",
        "",
        "## Decile Contrast (Risk Buckets)",
        f"- Highest-risk decile event rate: `{top_dec['event_rate']:.2%}` (lift `{top_dec['lift']:.2f}x`)",
        f"- Lowest-risk decile event rate: `{last_dec['event_rate']:.2%}` (lift `{last_dec['lift']:.2f}x`)",
        "",
        "## Threshold Tradeoff",
        threshold_text,
        "",
        "## Operational Recommendation",
        "- Use score ranking as the primary workflow: review the top risk bucket first.",
        "- If team capacity is fixed near 10%, the current model already gives meaningful enrichment versus random review.",
        "- If you need a hard flag, use the selected threshold metrics above to tune precision/recall based on team tolerance for false alarms.",
        "",
        "## Generated Artifacts",
        "- `results/stage1/plots/04_test_decile_event_rate.png`",
        "- `results/stage1/plots/05_test_capture_vs_workload.png`",
        "- `results/stage1/reports/business_impact_summary.md`",
    ]
    (REPORTS_DIR / "business_impact_summary.md").write_text("\n".join(lines))


def main() -> None:
    """
    Orchestrate all visualization generation.

    Reads walk-forward metrics and prediction files, creates all three plots,
    and writes them to `results/stage1/plots/` + `results/stage1/reports/`.
    """
    sns.set_theme(style="whitegrid", context="talk")

    wf = load_walk_forward()
    plot_top_decile_lift_by_fold(wf)
    plot_fold_by_fold_roc_pr(wf)
    plot_test_cumulative_lift_curve()
    best_model = get_best_classifier_name(default=BEST_TEST_MODEL)
    test_df = load_prediction_df(best_model, split="test")
    deciles = plot_test_decile_event_rate(test_df, best_model)
    capture = plot_test_capture_vs_workload(test_df, best_model)
    write_business_impact_summary(best_model, test_df, deciles, capture)

    print(f"Saved plots to: {OUT_DIR}")
    print(f"Saved business summary to: {REPORTS_DIR / 'business_impact_summary.md'}")


if __name__ == "__main__":
    main()
