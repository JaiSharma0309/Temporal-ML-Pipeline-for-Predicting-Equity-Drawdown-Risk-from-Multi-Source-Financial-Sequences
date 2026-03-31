from pathlib import Path
import itertools
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

DATA_PATH = Path("data/processed/stage1_modeling_data.csv")
OUT_DIR = Path("results/stage1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "label_drawdown_20pct_60d"
DATE_COL = "date"

TRAIN_END = "2021-12-31"
VAL_START = "2022-01-01"
VAL_END = "2023-12-31"
TEST_START = "2024-01-01"

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "volatility_20d",
    "volatility_60d",
    "downside_vol_60d",
    "dist_from_60d_high",
    "dist_from_252d_high",
    "trailing_max_drawdown_60d",
    "trailing_max_drawdown_252d",
    "avg_volume_20d",
    "avg_volume_60d",
    "volume_spike_20d",
    "avg_dollar_volume_20d",
    "mkt_ret_1d",
    "mkt_ret_20d",
    "mkt_ret_60d",
    "mkt_ret_120d",
    "mkt_volatility_20d",
    "mkt_volatility_60d",
    "rel_ret_20d_vs_mkt",
    "rel_ret_60d_vs_mkt",
    "rel_ret_120d_vs_mkt",
    "vol_ratio_20d_vs_mkt",
    "vol_ratio_60d_vs_mkt",
]

CATEGORICAL_FEATURES = [
    "country",
    "sector",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL, DATE_COL, "symbol", "country"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df.sort_values(DATE_COL).reset_index(drop=True)


def split_data(df: pd.DataFrame):
    train_mask = df[DATE_COL] <= pd.Timestamp(TRAIN_END)
    val_mask = (df[DATE_COL] >= pd.Timestamp(VAL_START)) & (df[DATE_COL] <= pd.Timestamp(VAL_END))
    test_mask = df[DATE_COL] >= pd.Timestamp(TEST_START)

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()
    test_df = df.loc[test_mask].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("One of the train/validation/test splits is empty.")

    return train_df, val_df, test_df


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    if scale_numeric:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def build_dummy_model():
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(scale_numeric=False)),
            ("model", DummyClassifier(strategy="prior")),
        ]
    )


def build_logistic_model():
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]
    )


def build_random_forest_model(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    max_features="sqrt",
):
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(scale_numeric=False)),
            ("model", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]
    )


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def evaluate_predictions(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": safe_auc(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def score_model(model, X, y):
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = y_pred.astype(float)

    metrics = evaluate_predictions(y, y_pred, y_score)
    return metrics, y_pred, y_score


def evaluate_at_threshold(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    return evaluate_predictions(y_true, y_pred, y_score)


def threshold_sweep(y_true, y_score, model_name, split_name, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.55, 0.05)

    rows = []
    for threshold in thresholds:
        metrics = evaluate_at_threshold(y_true, y_score, threshold)
        metrics["model"] = model_name
        metrics["split"] = split_name
        metrics["threshold"] = float(threshold)
        rows.append(metrics)

    return pd.DataFrame(rows)


def top_k_event_rate(y_true, y_score, k_frac=0.10):
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)

    n = len(y_true_arr)
    k = max(1, int(np.ceil(n * k_frac)))

    order = np.argsort(-y_score_arr)
    top_idx = order[:k]

    top_rate = float(np.mean(y_true_arr[top_idx]))
    base_rate = float(np.mean(y_true_arr))
    lift = float(top_rate / base_rate) if base_rate > 0 else np.nan

    return {
        "k_frac": float(k_frac),
        "top_k_n": int(k),
        "top_k_event_rate": top_rate,
        "base_rate": base_rate,
        "lift": lift,
    }


def save_predictions(df_split, model_name, split_name, y_true, y_pred, y_score):
    out = df_split[["date", "symbol", "country"]].copy()
    out["y_true"] = np.asarray(y_true)
    out["y_pred"] = np.asarray(y_pred)
    out["y_score"] = np.asarray(y_score)

    outpath = OUT_DIR / f"{model_name}_{split_name}_predictions.csv"
    out.to_csv(outpath, index=False)


def save_logistic_coefficients(model_pipeline):
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefs = model.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False)

    coef_df.to_csv(OUT_DIR / "logistic_regression_coefficients.csv", index=False)


def save_rf_importances(model_pipeline, filename="random_forest_feature_importances.csv"):
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    imp_df.to_csv(OUT_DIR / filename, index=False)


def get_rf_param_grid():
    grid = {
        "n_estimators": [200, 400],
        "max_depth": [4, 8, 12],
        "min_samples_leaf": [10, 25],
        "max_features": ["sqrt", 0.5],
    }

    keys = list(grid.keys())
    values = list(grid.values())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combos


def tune_random_forest(X_train, y_train, X_val, y_val):
    rows = []
    best_pipeline = None
    best_params = None
    best_score = -np.inf
    best_roc = -np.inf

    param_grid = get_rf_param_grid()

    print(f"Tuning random forest over {len(param_grid)} parameter combinations...")

    for i, params in enumerate(param_grid, start=1):
        print(f"  RF combo {i}/{len(param_grid)}: {params}")

        pipeline = build_random_forest_model(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
        )
        pipeline.fit(X_train, y_train)

        train_metrics, _, train_y_score = score_model(pipeline, X_train, y_train)
        val_metrics, _, val_y_score = score_model(pipeline, X_val, y_val)

        row = {
            **params,
            "train_pr_auc": train_metrics["pr_auc"],
            "train_roc_auc": train_metrics["roc_auc"],
            "validation_pr_auc": val_metrics["pr_auc"],
            "validation_roc_auc": val_metrics["roc_auc"],
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_metrics["f1"],
            "train_positive_score_mean": float(np.mean(train_y_score)),
            "validation_positive_score_mean": float(np.mean(val_y_score)),
        }
        rows.append(row)

        current_score = val_metrics["pr_auc"]
        current_roc = val_metrics["roc_auc"]

        if (current_score > best_score) or (current_score == best_score and current_roc > best_roc):
            best_score = current_score
            best_roc = current_roc
            best_pipeline = pipeline
            best_params = params

    results_df = pd.DataFrame(rows).sort_values(
        ["validation_pr_auc", "validation_roc_auc"],
        ascending=False
    ).reset_index(drop=True)

    results_df.to_csv(OUT_DIR / "random_forest_tuning_results.csv", index=False)

    return best_pipeline, best_params, results_df


def main():
    df = load_data()
    train_df, val_df, test_df = split_data(df)

    print("Split sizes:")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")
    print()

    for name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        pos_rate = split_df[TARGET_COL].mean()
        print(f"{name.title()} positive rate: {pos_rate:.4f}")
    print()

    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET_COL].astype(int)

    X_val = val_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_val = val_df[TARGET_COL].astype(int)

    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df[TARGET_COL].astype(int)

    fitted_models = {}
    all_metrics = []

    basic_models = {
        "dummy": build_dummy_model(),
        "logistic_regression": build_logistic_model(),
    }

    for model_name, pipeline in basic_models.items():
        print(f"Training {model_name}...")
        pipeline.fit(X_train, y_train)
        fitted_models[model_name] = pipeline

        for split_name, X_split, y_split, df_split in [
            ("train", X_train, y_train, train_df),
            ("validation", X_val, y_val, val_df),
            ("test", X_test, y_test, test_df),
        ]:
            metrics, y_pred, y_score = score_model(pipeline, X_split, y_split)
            metrics["model"] = model_name
            metrics["split"] = split_name
            all_metrics.append(metrics)

            save_predictions(df_split, model_name, split_name, y_split, y_pred, y_score)

            roc_text = "nan" if np.isnan(metrics["roc_auc"]) else f"{metrics['roc_auc']:.4f}"
            print(
                f"  {split_name:10s} "
                f"PR AUC={metrics['pr_auc']:.4f} "
                f"ROC AUC={roc_text}"
            )

        print()

    rf_pipeline, rf_best_params, rf_tuning_df = tune_random_forest(X_train, y_train, X_val, y_val)
    fitted_models["random_forest"] = rf_pipeline

    print()
    print("Best random forest parameters:")
    print(rf_best_params)
    print()

    for split_name, X_split, y_split, df_split in [
        ("train", X_train, y_train, train_df),
        ("validation", X_val, y_val, val_df),
        ("test", X_test, y_test, test_df),
    ]:
        metrics, y_pred, y_score = score_model(rf_pipeline, X_split, y_split)
        metrics["model"] = "random_forest"
        metrics["split"] = split_name
        all_metrics.append(metrics)

        save_predictions(df_split, "random_forest", split_name, y_split, y_pred, y_score)

        roc_text = "nan" if np.isnan(metrics["roc_auc"]) else f"{metrics['roc_auc']:.4f}"
        print(
            f"  random_forest {split_name:10s} "
            f"PR AUC={metrics['pr_auc']:.4f} "
            f"ROC AUC={roc_text}"
        )

    print()

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df[
        ["model", "split", "precision", "recall", "f1", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"]
    ]
    metrics_df.to_csv(OUT_DIR / "stage1_metrics.csv", index=False)

    val_metrics = metrics_df[metrics_df["split"] == "validation"].copy()
    best_model_name = val_metrics.sort_values("pr_auc", ascending=False).iloc[0]["model"]

    with open(OUT_DIR / "best_model.txt", "w") as f:
        f.write(str(best_model_name))

    print("Validation metrics:")
    print(val_metrics.sort_values("pr_auc", ascending=False).to_string(index=False))
    print()
    print(f"Best model by validation PR AUC: {best_model_name}")

    best_test_metrics = metrics_df[
        (metrics_df["model"] == best_model_name) & (metrics_df["split"] == "test")
    ]
    print()
    print("Best model test metrics at default threshold:")
    print(best_test_metrics.to_string(index=False))

    if "logistic_regression" in fitted_models:
        save_logistic_coefficients(fitted_models["logistic_regression"])

    if "random_forest" in fitted_models:
        save_rf_importances(fitted_models["random_forest"], filename="random_forest_feature_importances.csv")

    if best_model_name == "random_forest":
        best_model = fitted_models[best_model_name]

        val_y_score = best_model.predict_proba(X_val)[:, 1]
        test_y_score = best_model.predict_proba(X_test)[:, 1]

        val_threshold_df = threshold_sweep(y_val, val_y_score, best_model_name, "validation")
        test_threshold_df = threshold_sweep(y_test, test_y_score, best_model_name, "test")

        val_threshold_df.to_csv(OUT_DIR / "random_forest_validation_threshold_sweep.csv", index=False)
        test_threshold_df.to_csv(OUT_DIR / "random_forest_test_threshold_sweep.csv", index=False)

        best_threshold_row = val_threshold_df.sort_values("f1", ascending=False).iloc[0]
        selected_threshold = float(best_threshold_row["threshold"])

        with open(OUT_DIR / "random_forest_selected_threshold.txt", "w") as f:
            f.write(str(selected_threshold))

        final_test_metrics = evaluate_at_threshold(y_test, test_y_score, selected_threshold)
        final_test_df = pd.DataFrame([{
            "model": best_model_name,
            "split": "test_selected_threshold",
            "threshold": selected_threshold,
            **final_test_metrics
        }])
        final_test_df.to_csv(OUT_DIR / "random_forest_test_selected_threshold_metrics.csv", index=False)

        val_top10 = top_k_event_rate(y_val, val_y_score, k_frac=0.10)
        test_top10 = top_k_event_rate(y_test, test_y_score, k_frac=0.10)

        lift_df = pd.DataFrame([
            {"split": "validation", **val_top10},
            {"split": "test", **test_top10},
        ])
        lift_df.to_csv(OUT_DIR / "random_forest_top10pct_lift.csv", index=False)

        print()
        print(f"Selected random forest threshold from validation F1: {selected_threshold:.2f}")
        print("Validation threshold sweep top rows:")
        print(val_threshold_df.sort_values("f1", ascending=False).head(10).to_string(index=False))

        print()
        print("Test metrics at selected threshold:")
        print(final_test_df.to_string(index=False))

        print()
        print("Top 10% risk bucket lift:")
        print(lift_df.to_string(index=False))

        print()
        print("Top tuned RF validation rows:")
        print(rf_tuning_df.head(10).to_string(index=False))

    print()
    print(f"Saved metrics and predictions to: {OUT_DIR}")


if __name__ == "__main__":
    main()