from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/processed/stage1_modeling_data.csv")
OUT_DIR   = Path("results/stage1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "label_drawdown_20pct_60d"
DATE_COL   = "date"

# ── temporal split boundaries ──────────────────────────────────────────────────
TRAIN_END  = "2021-12-31"
VAL_START  = "2022-01-01"
VAL_END    = "2023-12-31"
TEST_START = "2024-01-01"

RANDOM_STATE = 42

# ── FIX 1: label boundary embargo ─────────────────────────────────────────────
# Training rows in the last ~3 months before TRAIN_END have labels computed
# from Jan–Mar 2022 prices (the start of the 2022 bear market).  The model
# learns "late-2021 price features → high drawdown risk" for the wrong reason
# (a macro rate-shock, not anything visible in price structure).  Dropping
# those rows removes this boundary leakage.
# ─────────────────────────────────────────────────────────────────────────────
LABEL_WINDOW_TRADING_DAYS = 60
EMBARGO_CALENDAR_DAYS     = 90   # ≈ 60 trading days

# ── FIX 2: purging overlapping label windows ───────────────────────────────────
# Consecutive daily rows for the same stock share 59/60 label-window days,
# so they are nearly identical in both features AND label.  The RF treats
# all 519k rows as independent — massively inflating apparent training fit.
# We thin to 1 row per PURGE_STRIDE days per stock.
#   stride = 60  →  fully non-overlapping (fewest rows, most correct)
#   stride = 20  →  keeps 3× more rows, residual ~40-day overlap (practical)
# ─────────────────────────────────────────────────────────────────────────────
PURGE_STRIDE = 20  # back to this

# ── feature flags ──────────────────────────────────────────────────────────────
# FIX 3: cross-sectional rank features  (regime-invariant signals)
# FIX 4: walk-forward CV                (honest multi-regime evaluation)
USE_RANK_FEATURES   = True
RUN_WALK_FORWARD_CV = True

# ── RF hyperparameters ────────────────────────────────────────────────────────
# Re-tuned for the purged dataset (~26k rows at stride=20 vs 519k raw).
# min_samples_leaf / min_samples_split were too large for the new effective N.
RF_N_ESTIMATORS      = 100
# RF — heavier regularisation
RF_MAX_DEPTH         = 6    # was 10, key lever
RF_MIN_SAMPLES_LEAF  = 30   # was 20
RF_MIN_SAMPLES_SPLIT = 150  # was 100
RF_MAX_FEATURES      = 0.2  # was 0.3, see fewer features per split

# ── HGB hyperparameters ────────────────────────────────────────────────────────
HGB_LEARNING_RATE    = 0.05
HGB_MAX_ITER         = 200   # was 150; bumped because early_stopping disabled (see FIX 5)
HGB_MAX_DEPTH        = 4    # was 5
HGB_MIN_SAMPLES_LEAF = 100  # was 50
HGB_L2               = 5.0  # was 2.0
HGB_MAX_ITER         = 150  # back to 150

# ── base feature lists ─────────────────────────────────────────────────────────
BASE_NUMERIC_FEATURES = [
    "ret_1d", "ret_5d", "ret_20d", "ret_60d", "ret_120d",
    "volatility_20d", "volatility_60d",
    "downside_vol_60d", "upside_vol_60d", "downside_upside_vol_ratio_60d",
    "dist_from_60d_high", "dist_from_252d_high",
    "days_since_60d_high", "days_since_252d_high",
    "trailing_max_drawdown_60d", "trailing_max_drawdown_252d",
    "avg_volume_20d", "avg_volume_60d", "volume_spike_20d", "avg_dollar_volume_20d",
    "price_to_ma20", "price_to_ma60", "price_to_ma120",
    "ma20_to_ma60", "ma60_to_ma120",
    "log_price_slope_20d", "log_price_slope_60d", "log_price_slope_120d",
    "trend_accel_20_60",
    "return_skew_20d", "return_skew_60d",
    "gap_open_prev_close", "intraday_range", "avg_intraday_range_20d",
    "mkt_ret_1d", "mkt_ret_20d", "mkt_ret_60d", "mkt_ret_120d",
    "mkt_volatility_20d", "mkt_volatility_60d",
    "rel_ret_20d_vs_mkt", "rel_ret_60d_vs_mkt", "rel_ret_120d_vs_mkt",
    "vol_ratio_20d_vs_mkt", "vol_ratio_60d_vs_mkt",
    "rel_drawdown_60d_vs_mkt", "rel_drawdown_252d_vs_mkt",
    "rel_slope_20d_vs_mkt", "rel_slope_60d_vs_mkt",
    "beta_60d_vs_mkt", "corr_60d_vs_mkt", "idio_vol_60d_vs_mkt",
    "sector_ret_1d", "sector_ret_20d", "sector_ret_60d", "sector_ret_120d",
    "sector_volatility_20d", "sector_volatility_60d",
    "rel_ret_20d_vs_sector", "rel_ret_60d_vs_sector", "rel_ret_120d_vs_sector",
    "vol_ratio_20d_vs_sector", "vol_ratio_60d_vs_sector",
    "rel_drawdown_60d_vs_sector", "rel_drawdown_252d_vs_sector",
    "rel_slope_20d_vs_sector", "rel_slope_60d_vs_sector",
    "beta_60d_vs_sector", "corr_60d_vs_sector", "idio_vol_60d_vs_sector",
]

# Features to generate cross-sectional percentile ranks for.
# Absolute price/momentum features are the most regime-sensitive: ret_60d=+15%
# means very different things in a 2019 bull market vs a 2022 bear market.
# Ranking within the cross-section on each date produces a signal that is
# comparably meaningful regardless of the market environment.
FEATURES_TO_RANK = [
    "ret_20d", "ret_60d", "ret_120d",
    "volatility_60d", "downside_vol_60d",
    "dist_from_60d_high", "dist_from_252d_high",
    "trailing_max_drawdown_60d", "trailing_max_drawdown_252d",
    "price_to_ma60", "price_to_ma120",
    "log_price_slope_60d", "log_price_slope_120d",
    "rel_ret_60d_vs_mkt", "rel_ret_120d_vs_mkt",
    "rel_drawdown_60d_vs_mkt", "rel_drawdown_252d_vs_mkt",
    "idio_vol_60d_vs_mkt",
    "rel_ret_60d_vs_sector", "rel_drawdown_60d_vs_sector",
]

CATEGORICAL_FEATURES = ["country", "sector"]


# ══════════════════════════════════════════════════════════════════════════════
# FIX 3 — Regime features & cross-sectional rank features
# ══════════════════════════════════════════════════════════════════════════════

def add_regime_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Compute simple market-regime binary flags from existing backward-looking
    market features.  The same price-based predictors have different relationships
    with forward drawdowns depending on whether the market is in a bull or bear
    regime.  These flags let the model condition its predictions on the regime
    without needing external data (e.g. VIX).

    Safe to compute on the full df before splitting — each flag only uses
    columns that are backward-looking at time T, so no lookahead.
    """
    df = df.copy()

    # Is the broad market in a medium-term uptrend?
    df["regime_mkt_uptrend"] = (df["mkt_ret_120d"] > 0).astype(float)

    # Is the broad market under near-term stress?
    df["regime_mkt_stress"] = (df["mkt_ret_20d"] < -0.05).astype(float)

    # Is broad-market volatility elevated (rough fear proxy)?
    # Use a rolling 70th-percentile threshold computed over the full sample.
    vol_threshold = df["mkt_volatility_60d"].quantile(0.70)
    df["regime_vol_elevated"] = (df["mkt_volatility_60d"] > vol_threshold).astype(float)

    regime_cols = ["regime_mkt_uptrend", "regime_mkt_stress", "regime_vol_elevated"]
    return df, regime_cols


def add_cross_sectional_ranks(
    df: pd.DataFrame,
    features: list,
    date_col: str = DATE_COL,
) -> tuple[pd.DataFrame, list]:
    """
    For each feature in `features`, compute the cross-sectional percentile rank
    of each stock within the universe on that date (0 = lowest, 1 = highest).

    Why: absolute values of momentum/volatility features shift dramatically
    across market regimes, making them poor out-of-sample predictors when the
    regime in val/test differs from training.  Percentile ranks within the
    cross-section are stable across regimes.

    No lookahead: ranks are computed groupby date, so each date's ranks only
    use stocks available on that same date.  Safe to run on the full df before
    the train/val/test split.
    """
    df = df.copy()
    rank_cols = []
    for col in features:
        if col in df.columns:
            rank_col = f"{col}_xsrank"
            df[rank_col] = df.groupby(date_col)[col].rank(pct=True, na_option="keep")
            rank_cols.append(rank_col)
    return df, rank_cols


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prepare_data() -> tuple[pd.DataFrame, list]:
    """
    Load the CSV, engineer regime + rank features, validate columns.
    Returns (enriched_df, all_numeric_features).
    All feature engineering runs on the full df before any split so that
    cross-sectional ranks are computed consistently across the entire universe.
    """
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    df, regime_cols = add_regime_features(df)

    rank_cols = []
    if USE_RANK_FEATURES:
        df, rank_cols = add_cross_sectional_ranks(df, FEATURES_TO_RANK)

    all_numeric = BASE_NUMERIC_FEATURES + regime_cols + rank_cols

    required = set(all_numeric + CATEGORICAL_FEATURES + [TARGET_COL, DATE_COL, "symbol"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after feature engineering: {missing}")

    return df, all_numeric


# ══════════════════════════════════════════════════════════════════════════════
# FIX 1 + FIX 2 — Embargo & purging helpers
# ══════════════════════════════════════════════════════════════════════════════

def apply_train_embargo(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX 1: Drop the final EMBARGO_CALENDAR_DAYS of training data.
    Rows in Q4 2021 have labels built from Jan–Mar 2022 prices.
    See module-level comment for full explanation.
    """
    cutoff = pd.Timestamp(TRAIN_END) - pd.Timedelta(days=EMBARGO_CALENDAR_DAYS)
    kept = df[df[DATE_COL] <= cutoff].copy()
    n_dropped = len(df) - len(kept)
    print(f"  [embargo]  dropped {n_dropped:,} rows from last {EMBARGO_CALENDAR_DAYS} calendar days of train")
    return kept


def purge_overlapping_labels(df: pd.DataFrame, stride: int = PURGE_STRIDE) -> pd.DataFrame:
    """
    FIX 2: Subsample training rows so consecutive kept rows per stock are
    `stride` trading days apart, dramatically reducing serial correlation from
    overlapping 60-day label windows.
    Applied to TRAINING ONLY — val/test are evaluated on all their rows.
    """
    df = df.copy().sort_values([DATE_COL, "symbol"])
    df["_cumcount"] = df.groupby("symbol").cumcount()
    purged = df[df["_cumcount"] % stride == 0].drop(columns=["_cumcount"])
    n_dropped = len(df) - len(purged)
    print(f"  [purge]    stride={stride}: {len(purged):,} rows kept, {n_dropped:,} dropped "
          f"({100 * n_dropped / len(df):.0f}% removed)")
    return purged.reset_index(drop=True)


def split_data(df: pd.DataFrame):
    """
    Temporal split with embargo + purge applied to training only.
    Val and test are returned as-is (full daily frequency).
    """
    train_mask = df[DATE_COL] <= pd.Timestamp(TRAIN_END)
    val_mask   = (df[DATE_COL] >= pd.Timestamp(VAL_START)) & (df[DATE_COL] <= pd.Timestamp(VAL_END))
    test_mask  = df[DATE_COL] >= pd.Timestamp(TEST_START)

    train_df = df[train_mask].copy()
    val_df   = df[val_mask].copy()
    test_df  = df[test_mask].copy()

    print("Applying embargo + purge to training data:")
    train_df = apply_train_embargo(train_df)
    train_df = purge_overlapping_labels(train_df)

    for name, split in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        if len(split) == 0:
            raise ValueError(f"{name} split is empty after preprocessing.")

    return train_df, val_df, test_df


# ══════════════════════════════════════════════════════════════════════════════
# FIX 4 — Walk-forward cross-validation
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_splits(
    df: pd.DataFrame,
    initial_train_years: int = 3,
    val_years: int = 1,
    step_years: int = 1,
) -> list:
    """
    Generate expanding-window (train_df, val_df, label) fold tuples.
    Each fold applies the same embargo + purge as the main split, so fold
    metrics are honest estimates of out-of-sample performance across multiple
    distinct market regimes.

    With data starting ~2017 and initial_train_years=3:
      Fold 1: train 2017–2019  →  val 2020  (COVID crash)
      Fold 2: train 2017–2020  →  val 2021  (bull recovery)
      Fold 3: train 2017–2021  →  val 2022  (rate-hike bear)
      Fold 4: train 2017–2022  →  val 2023  (mixed)
    """
    df = df.sort_values(DATE_COL)
    min_date = df[DATE_COL].min()

    folds = []
    train_end_dt = min_date + pd.DateOffset(years=initial_train_years)

    while True:
        val_end_dt = train_end_dt + pd.DateOffset(years=val_years)
        if val_end_dt > df[DATE_COL].max():
            break

        effective_train_end = train_end_dt - pd.Timedelta(days=EMBARGO_CALENDAR_DAYS)

        train_fold = df[df[DATE_COL] <= effective_train_end].copy()
        # val starts strictly after train_end_dt to respect the label window gap
        val_fold   = df[(df[DATE_COL] > train_end_dt) & (df[DATE_COL] <= val_end_dt)].copy()

        if len(train_fold) > 0 and len(val_fold) > 0:
            train_fold = purge_overlapping_labels(train_fold, stride=PURGE_STRIDE)
            fold_label = (
                f"val_{train_end_dt.year}–{int(val_end_dt.year) - 1}"
                if val_end_dt.year > train_end_dt.year
                else f"val_{train_end_dt.year}"
            )
            folds.append((train_fold, val_fold, fold_label))

        train_end_dt += pd.DateOffset(years=step_years)

    return folds


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessors & model builders
# ══════════════════════════════════════════════════════════════════════════════

def make_standard_preprocessor(numeric_features: list, scale_numeric: bool) -> ColumnTransformer:
    if scale_numeric:
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
    else:
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_transformer,   numeric_features),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def make_histgb_preprocessor(numeric_features: list) -> ColumnTransformer:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )),
    ])

    return ColumnTransformer([
        ("num", numeric_transformer,     numeric_features),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def build_dummy_model(numeric_features: list) -> Pipeline:
    return Pipeline([
        ("preprocessor", make_standard_preprocessor(numeric_features, scale_numeric=False)),
        ("model",        DummyClassifier(strategy="prior")),
    ])


def build_logistic_model(numeric_features: list) -> Pipeline:
    return Pipeline([
        ("preprocessor", make_standard_preprocessor(numeric_features, scale_numeric=True)),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def build_random_forest_model(numeric_features: list) -> Pipeline:
    return Pipeline([
        ("preprocessor", make_standard_preprocessor(numeric_features, scale_numeric=False)),
        ("model", RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            max_features=RF_MAX_FEATURES,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def build_hist_gradient_boosting_model(numeric_features: list) -> Pipeline:
    # FIX 5: early_stopping was set to True with a random internal validation
    # fraction drawn from training data.  That internal val set has the same
    # overlapping-label serial correlation as training, so early stopping
    # fired based on a leaky signal and is not a reliable stopping criterion.
    # Disabled here; use a conservative fixed max_iter instead.
    categorical_feature_indices = list(
        range(len(numeric_features), len(numeric_features) + len(CATEGORICAL_FEATURES))
    )

    return Pipeline([
        ("preprocessor", make_histgb_preprocessor(numeric_features)),
        ("model", HistGradientBoostingClassifier(
            learning_rate=HGB_LEARNING_RATE,
            max_iter=HGB_MAX_ITER,
            max_depth=HGB_MAX_DEPTH,
            min_samples_leaf=HGB_MIN_SAMPLES_LEAF,
            l2_regularization=HGB_L2,
            early_stopping=False,          # FIX 5: was True with random val fraction
            categorical_features=categorical_feature_indices,
            random_state=RANDOM_STATE,
        )),
    ])


def get_models(numeric_features: list) -> dict:
    return {
        "dummy":                  build_dummy_model(numeric_features),
        "logistic_regression":    build_logistic_model(numeric_features),
        "random_forest":          build_random_forest_model(numeric_features),
        "hist_gradient_boosting": build_hist_gradient_boosting_model(numeric_features),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def evaluate_predictions(y_true, y_pred, y_score) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   safe_auc(y_true, y_score),
        "pr_auc":    average_precision_score(y_true, y_score),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def score_model(model, X, y) -> tuple:
    y_pred  = model.predict(X)
    y_score = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)
    return evaluate_predictions(y, y_pred, y_score), y_pred, y_score


def evaluate_at_threshold(y_true, y_score, threshold) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return evaluate_predictions(y_true, y_pred, y_score)


def threshold_sweep(y_true, y_score, model_name, split_name, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.05, 0.55, 0.05)
    rows = []
    for t in thresholds:
        m = evaluate_at_threshold(y_true, y_score, t)
        m["model"] = model_name
        m["split"] = split_name
        m["threshold"] = float(t)
        rows.append(m)
    return pd.DataFrame(rows)


def top_k_event_rate(y_true, y_score, k_frac=0.10) -> dict:
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    k       = max(1, int(np.ceil(len(y_true) * k_frac)))
    top_idx = np.argsort(-y_score)[:k]
    top_rate  = float(np.mean(y_true[top_idx]))
    base_rate = float(np.mean(y_true))
    return {
        "k_frac":           float(k_frac),
        "top_k_n":          int(k),
        "top_k_event_rate": top_rate,
        "base_rate":        base_rate,
        "lift":             float(top_rate / base_rate) if base_rate > 0 else np.nan,
    }


def save_predictions(df_split, model_name, split_name, y_true, y_pred, y_score):
    out = df_split[["date", "symbol", "country", "sector"]].copy()
    out["y_true"]  = np.asarray(y_true)
    out["y_pred"]  = np.asarray(y_pred)
    out["y_score"] = np.asarray(y_score)
    out.to_csv(OUT_DIR / f"{model_name}_{split_name}_predictions.csv", index=False)


def save_logistic_coefficients(pipeline):
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    pd.DataFrame({
        "feature":         feature_names,
        "coefficient":     coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False).to_csv(
        OUT_DIR / "logistic_regression_coefficients.csv", index=False
    )


def save_rf_importances(pipeline):
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances   = pipeline.named_steps["model"].feature_importances_
    pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).to_csv(
        OUT_DIR / "random_forest_feature_importances.csv", index=False
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIX 4 — Walk-forward CV runner
# ══════════════════════════════════════════════════════════════════════════════

def run_walk_forward_cv(df: pd.DataFrame, numeric_features: list) -> pd.DataFrame:
    """
    Train each model on each walk-forward fold and report per-fold + averaged
    val metrics.  This gives a much more honest picture of generalisation
    than a single train/val split, because the model is evaluated against
    multiple distinct market regimes (Covid crash, bull recovery, rate-hike
    bear, etc.).
    """
    print("\n" + "=" * 70)
    print("WALK-FORWARD CROSS-VALIDATION")
    print("=" * 70)

    folds = walk_forward_splits(df)
    if not folds:
        print("  Not enough data for walk-forward CV (need > initial_train_years + val_years).")
        return pd.DataFrame()

    # Skip dummy in CV — it always gives the same result
    model_names = ["logistic_regression", "random_forest", "hist_gradient_boosting"]
    all_rows = []

    for fold_idx, (train_fold, val_fold, fold_label) in enumerate(folds, 1):
        print(f"\nFold {fold_idx}: {fold_label}  "
              f"(train={len(train_fold):,} rows, val={len(val_fold):,} rows, "
              f"val_pos_rate={val_fold[TARGET_COL].mean():.4f})")

        X_tr = train_fold[numeric_features + CATEGORICAL_FEATURES]
        y_tr = train_fold[TARGET_COL].astype(int)
        X_vl = val_fold[numeric_features + CATEGORICAL_FEATURES]
        y_vl = val_fold[TARGET_COL].astype(int)

        for model_name in model_names:
            models = get_models(numeric_features)
            pipe   = models[model_name]
            pipe.fit(X_tr, y_tr)

            metrics, _, y_score = score_model(pipe, X_vl, y_vl)
            lift = top_k_event_rate(y_vl, y_score, k_frac=0.10)

            row = {
                "fold":       fold_label,
                "model":      model_name,
                "val_roc_auc": metrics["roc_auc"],
                "val_pr_auc":  metrics["pr_auc"],
                "top10_lift":  lift["lift"],
                "val_pos_rate": float(y_vl.mean()),
                "val_n":       int(len(y_vl)),
            }
            all_rows.append(row)
            print(f"  {model_name:25s}  ROC={metrics['roc_auc']:.4f}  "
                  f"PR={metrics['pr_auc']:.4f}  lift@10%={lift['lift']:.2f}x")

    cv_df = pd.DataFrame(all_rows)

    print("\nWalk-forward CV — averages across folds:")
    avg = (
        cv_df.groupby("model")[["val_roc_auc", "val_pr_auc", "top10_lift"]]
        .mean()
        .sort_values("val_roc_auc", ascending=False)
    )
    print(avg.round(4).to_string())

    cv_df.to_csv(OUT_DIR / "walk_forward_cv_results.csv", index=False)
    print(f"\nSaved walk-forward CV results to: {OUT_DIR / 'walk_forward_cv_results.csv'}")
    print("=" * 70)

    return cv_df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── load & engineer features ───────────────────────────────────────────────
    print("Loading data and engineering features...")
    df, numeric_features = load_and_prepare_data()
    print(f"  Loaded {len(df):,} rows | {len(numeric_features)} numeric features "
          f"({len(BASE_NUMERIC_FEATURES)} base + regime + "
          f"{'rank' if USE_RANK_FEATURES else 'no rank'})")

    # ── walk-forward CV (optional) ─────────────────────────────────────────────
    if RUN_WALK_FORWARD_CV:
        run_walk_forward_cv(df, numeric_features)

    # ── final train / val / test split ─────────────────────────────────────────
    print("\nBuilding final train / val / test split:")
    train_df, val_df, test_df = split_data(df)

    print("\nSplit sizes:")
    for name, split in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"  {name:12s}: {len(split):>8,} rows  |  pos rate: {split[TARGET_COL].mean():.4f}")

    print()
    print(f"Random Forest config:  depth={RF_MAX_DEPTH}, leaf={RF_MIN_SAMPLES_LEAF}, "
          f"split={RF_MIN_SAMPLES_SPLIT}, features={RF_MAX_FEATURES}")
    print(f"HGB config:            lr={HGB_LEARNING_RATE}, iter={HGB_MAX_ITER}, "
          f"depth={HGB_MAX_DEPTH}, leaf={HGB_MIN_SAMPLES_LEAF}, l2={HGB_L2}")
    print()

    X_train = train_df[numeric_features + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET_COL].astype(int)
    X_val   = val_df[numeric_features + CATEGORICAL_FEATURES]
    y_val   = val_df[TARGET_COL].astype(int)
    X_test  = test_df[numeric_features + CATEGORICAL_FEATURES]
    y_test  = test_df[TARGET_COL].astype(int)

    # ── train all models ───────────────────────────────────────────────────────
    models       = get_models(numeric_features)
    fitted_models = {}
    all_metrics  = []

    for model_name, pipeline in models.items():
        print(f"Training {model_name}...")
        pipeline.fit(X_train, y_train)
        fitted_models[model_name] = pipeline

        for split_name, X_split, y_split, df_split in [
            ("train",      X_train, y_train, train_df),
            ("validation", X_val,   y_val,   val_df),
            ("test",       X_test,  y_test,  test_df),
        ]:
            metrics, y_pred, y_score = score_model(pipeline, X_split, y_split)
            metrics["model"] = model_name
            metrics["split"] = split_name
            all_metrics.append(metrics)
            save_predictions(df_split, model_name, split_name, y_split, y_pred, y_score)

            roc_text = "nan" if np.isnan(metrics["roc_auc"]) else f"{metrics['roc_auc']:.4f}"
            print(f"  {split_name:10s}  PR AUC={metrics['pr_auc']:.4f}  ROC AUC={roc_text}")

        print()

    # ── model selection & reporting ────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)[
        ["model", "split", "precision", "recall", "f1", "roc_auc", "pr_auc",
         "tn", "fp", "fn", "tp"]
    ]
    metrics_df.to_csv(OUT_DIR / "stage1_metrics.csv", index=False)

    val_metrics = metrics_df[metrics_df["split"] == "validation"].copy()
    best_model_name = val_metrics.sort_values("pr_auc", ascending=False).iloc[0]["model"]

    with open(OUT_DIR / "best_model.txt", "w") as f:
        f.write(str(best_model_name))

    print("Validation metrics:")
    print(val_metrics.sort_values("pr_auc", ascending=False).to_string(index=False))
    print(f"\nBest model by validation PR AUC: {best_model_name}")

    best_test = metrics_df[(metrics_df["model"] == best_model_name) & (metrics_df["split"] == "test")]
    print("\nBest model test metrics at default threshold:")
    print(best_test.to_string(index=False))

    # ── feature importances ────────────────────────────────────────────────────
    if "logistic_regression" in fitted_models:
        save_logistic_coefficients(fitted_models["logistic_regression"])
    if "random_forest" in fitted_models:
        save_rf_importances(fitted_models["random_forest"])

    # ── threshold sweep + lift for best model ──────────────────────────────────
    best_model = fitted_models.get(best_model_name)
    if best_model and hasattr(best_model, "predict_proba"):
        val_y_score  = best_model.predict_proba(X_val)[:, 1]
        test_y_score = best_model.predict_proba(X_test)[:, 1]

        val_thr_df  = threshold_sweep(y_val,  val_y_score,  best_model_name, "validation")
        test_thr_df = threshold_sweep(y_test, test_y_score, best_model_name, "test")
        val_thr_df.to_csv(OUT_DIR  / f"{best_model_name}_validation_threshold_sweep.csv", index=False)
        test_thr_df.to_csv(OUT_DIR / f"{best_model_name}_test_threshold_sweep.csv",       index=False)

        best_thr_row       = val_thr_df.sort_values("f1", ascending=False).iloc[0]
        selected_threshold = float(best_thr_row["threshold"])

        with open(OUT_DIR / f"{best_model_name}_selected_threshold.txt", "w") as f:
            f.write(str(selected_threshold))

        final_test_metrics = evaluate_at_threshold(y_test, test_y_score, selected_threshold)
        final_test_df = pd.DataFrame([{
            "model":     best_model_name,
            "split":     "test_selected_threshold",
            "threshold": selected_threshold,
            **final_test_metrics,
        }])
        final_test_df.to_csv(OUT_DIR / f"{best_model_name}_test_selected_threshold_metrics.csv", index=False)

        val_lift  = top_k_event_rate(y_val,  val_y_score,  k_frac=0.10)
        test_lift = top_k_event_rate(y_test, test_y_score, k_frac=0.10)
        lift_df   = pd.DataFrame([
            {"split": "validation", **val_lift},
            {"split": "test",       **test_lift},
        ])
        lift_df.to_csv(OUT_DIR / f"{best_model_name}_top10pct_lift.csv", index=False)

        print(f"\nSelected {best_model_name} threshold from validation F1: {selected_threshold:.2f}")
        print("Validation threshold sweep top rows:")
        print(val_thr_df.sort_values("f1", ascending=False).head(10).to_string(index=False))

        print("\nTest metrics at selected threshold:")
        print(final_test_df.to_string(index=False))

        print("\nTop 10% risk bucket lift:")
        print(lift_df.to_string(index=False))

    print(f"\nSaved metrics and predictions to: {OUT_DIR}")


if __name__ == "__main__":
    main()