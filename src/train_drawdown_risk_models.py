"""
train_drawdown_risk_models.py
=============================
Author: Jai Sharma

Time-aware ML pipeline for predicting 60-day equity drawdown risk across
the S&P 500 + TSX 60 universe (2016–2025).

Architecture overview
---------------------
  1. Feature engineering
       • 70 price/volume/benchmark base features
       • 2 fixed-threshold regime flags  (regime_mkt_uptrend, regime_mkt_stress)
       • 1 fitted regime flag            (regime_vol_elevated — fitted in pipeline)
       • 20 cross-sectional rank features (regime-invariant)
       • 4 short-interest features       (optional — requires fetch_short_interest.py)
       • 7 fundamental features          (optional — requires fetch_fundamentals.py)

  2. Target columns
       • label_drawdown_20pct_60d   — binary  (classification)
       • future_drawdown_60d        — continuous (regression ranking)

  3. Temporal safeguards
       • Label boundary embargo: drop last 90 calendar days of training
         (those rows' labels use 2022 prices → leakage)
       • Overlapping-label purge: keep 1 row per 20 trading days per stock
         (consecutive rows share 59/60 label-window days → serial correlation)
       • Walk-forward CV: 4 expanding folds covering 2020/21/22/23

  4. Models
       Classification  — Dummy, Logistic Regression, Random Forest, HGB
       Regression      — Ridge, RF Regressor, HGB Regressor
         (trained on future_drawdown_60d; evaluated by ranking binary label)

Run:
    python train_drawdown_risk_models.py
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_absolute_error, r2_score,
)


# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

DATA_PATH          = REPO_ROOT / "data/processed/stage1_modeling_data.csv"
SI_PATH            = REPO_ROOT / "data/raw/short_interest/finra_short_interest_raw.parquet"
FUNDAMENTALS_PATH  = REPO_ROOT / "data/raw/fundamentals/fundamentals_features.parquet"
OUT_DIR            = REPO_ROOT / "results/stage1"
TABLES_DIR         = OUT_DIR / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Target columns
# ══════════════════════════════════════════════════════════════════════════════

TARGET_CLF = "label_drawdown_20pct_60d"   # binary  — used to train classifiers
TARGET_REG = "future_drawdown_60d"        # continuous (negative = drawdown)
                                          # used to train regressors;
                                          # risk score = –predicted_drawdown
DATE_COL   = "date"


# ══════════════════════════════════════════════════════════════════════════════
# Temporal split boundaries
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_END  = "2021-12-31"
VAL_START  = "2022-01-01"
VAL_END    = "2023-12-31"
TEST_START = "2024-01-01"

RANDOM_STATE = 42

# ── label boundary embargo ────────────────────────────────────────────────────
# Training rows in the last ~3 months before TRAIN_END have labels computed
# from Jan–Mar 2022 prices (the start of the 2022 bear market).  Dropping
# those rows removes boundary leakage.
EMBARGO_CALENDAR_DAYS = 90   # ≈ 60 trading days

# ── overlapping-label purging ─────────────────────────────────────────────────
# Consecutive daily rows share 59/60 label-window days.  Subsampling to 1 row
# per PURGE_STRIDE trading days per stock reduces serial correlation.
PURGE_STRIDE = 20


# ══════════════════════════════════════════════════════════════════════════════
# Feature flags
# ══════════════════════════════════════════════════════════════════════════════

USE_RANK_FEATURES    = True   # cross-sectional percentile ranks
USE_SHORT_INTEREST   = True   # requires SI_PATH to exist
USE_FUNDAMENTALS     = True   # requires FUNDAMENTALS_PATH to exist
RUN_WALK_FORWARD_CV  = True
RUN_REGRESSION_MODELS = True  # regression block alongside classification

# Logging mode (set from CLI in main)
VERBOSE = False


def vprint(*args, **kwargs):
    """
    Print only when verbose mode is enabled.

    @param args: Positional arguments forwarded to `print`.
    @param kwargs: Keyword arguments forwarded to `print`.
    @return: None.
    """
    if VERBOSE:
        print(*args, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

# ── Random Forest (classifier + regressor) ────────────────────────────────────
RF_N_ESTIMATORS      = 100
RF_MAX_DEPTH         = 6
RF_MIN_SAMPLES_LEAF  = 30
RF_MIN_SAMPLES_SPLIT = 150
RF_MAX_FEATURES      = 0.2

# ── HistGradientBoosting (classifier + regressor) ─────────────────────────────
HGB_LEARNING_RATE    = 0.05
HGB_MAX_ITER         = 150
HGB_MAX_DEPTH        = 4
HGB_MIN_SAMPLES_LEAF = 100
HGB_L2               = 5.0

# ── Logistic Regression ───────────────────────────────────────────────────────
# C=0.01 (100× more L2 than sklearn default).  Several feature pairs have
# r≈1.0 (ret_60d / rel_ret_60d_vs_mkt etc.) — strong L2 prevents LR from
# building large opposing coefficients that cancel in training but blow up OOS.
LR_C = 0.01

# ── Ridge (regression) ────────────────────────────────────────────────────────
RIDGE_ALPHA = 100.0   # analogous to 1/C — high = strong L2


# ══════════════════════════════════════════════════════════════════════════════
# Base feature lists
# ══════════════════════════════════════════════════════════════════════════════

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

# Absolute price/momentum features are regime-sensitive.  We add percentile
# ranks within the cross-section on each date alongside the raw values.
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

# Short interest feature names (populated by build_short_interest_features)
SI_FEATURE_COLS = [
    "short_interest_ratio",
    "si_chg_1period",
    "si_chg_3period",
    "sir_zscore_1y",
]

# Fundamental feature names (populated by build_fundamental_features)
FUNDAMENTAL_FEATURE_COLS = [
    "revenue_growth_yoy",
    "revenue_growth_decel",
    "gross_margin",
    "operating_margin",
    "debt_to_equity",
    "interest_coverage",
    "current_ratio",
]


# ══════════════════════════════════════════════════════════════════════════════
# Custom sklearn transformer — regime vol flag (fitted on training data only)
# ══════════════════════════════════════════════════════════════════════════════

class RegimeVolTransformer(BaseEstimator, TransformerMixin):
    """
    Converts mkt_volatility_60d into a binary 'elevated' flag.

    The 70th-percentile threshold is fitted on X_train only, so it never sees
    val/test distribution.  This fixes leakage that existed when the threshold
    was computed on the full dataset before the train/val split.

    Input:  (n, 1) array of vol values
    Output: (n, 1) float binary column  {0.0, 1.0}
    """
    def __init__(self, quantile: float = 0.70):
        """
        Initialize the volatility-threshold transformer.

        @param quantile: Quantile used to determine the elevated-volatility cutoff.
        @return: None.
        """
        self.quantile = quantile

    def fit(self, X, y=None):
        """
        Fit the volatility threshold from the input data.

        @param X: Input volatility values.
        @param y: Unused target array, present for sklearn compatibility.
        @return: Fitted transformer instance.
        """
        self.threshold_ = float(np.nanquantile(X, self.quantile))
        return self

    def transform(self, X):
        """
        Convert raw volatility values into an elevated-volatility flag.

        @param X: Input volatility values.
        @return: Binary numpy array indicating elevated volatility.
        """
        return (np.asarray(X) > self.threshold_).astype(float)

    def get_feature_names_out(self, input_features=None):
        """
        Return output feature names for sklearn compatibility.

        @param input_features: Optional input feature names.
        @return: Array containing the derived feature name.
        """
        return np.array(["regime_vol_elevated"])


# ══════════════════════════════════════════════════════════════════════════════
# Feature engineering — regime flags
# ══════════════════════════════════════════════════════════════════════════════

def add_regime_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Add fixed-threshold market regime binary flags.  These use only domain-
    knowledge thresholds (no quantile fitting) so they are safe to compute on
    the full df before the train/val split.

    regime_vol_elevated is handled separately inside each sklearn Pipeline via
    RegimeVolTransformer so it is fitted on X_train only.

    @param df: Modeling dataframe.
    @return: Tuple of augmented dataframe and newly added regime feature names.
    """
    df = df.copy()
    df["regime_mkt_uptrend"] = (df["mkt_ret_120d"] > 0).astype(float)
    df["regime_mkt_stress"]  = (df["mkt_ret_20d"]  < -0.05).astype(float)
    return df, ["regime_mkt_uptrend", "regime_mkt_stress"]


# ══════════════════════════════════════════════════════════════════════════════
# Feature engineering — cross-sectional ranks
# ══════════════════════════════════════════════════════════════════════════════

def add_cross_sectional_ranks(
    df: pd.DataFrame,
    features: list,
    date_col: str = DATE_COL,
) -> tuple[pd.DataFrame, list]:
    """
    For each feature, compute the percentile rank within the cross-section on
    each date (0 = lowest, 1 = highest).  No lookahead: ranks are computed
    groupby date, using only stocks present on that date.

    @param df: Modeling dataframe.
    @param features: Feature names to rank cross-sectionally.
    @param date_col: Date column used to define each cross-section.
    @return: Tuple of augmented dataframe and rank feature names.
    """
    df = df.copy()
    rank_cols = []
    for col in features:
        if col in df.columns:
            rc = f"{col}_xsrank"
            df[rc] = df.groupby(date_col)[col].rank(pct=True, na_option="keep")
            rank_cols.append(rc)
    return df, rank_cols


# ══════════════════════════════════════════════════════════════════════════════
# Feature engineering — short interest
# ══════════════════════════════════════════════════════════════════════════════

def build_short_interest_features(
    si_raw: pd.DataFrame,
    price_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list]:
    """
    Build per-stock daily short interest features by forward-filling the most
    recent available FINRA settlement report onto every daily price row.

    Features
    --------
    short_interest_ratio  — shares short / avg daily volume (days-to-cover)
                            High SIR = short squeeze risk AND informed pessimism
    si_chg_1period        — % change in shares short since 1 settlement ago (~15d)
                            Rapidly rising SI often precedes drawdowns
    si_chg_3period        — % change since 3 settlements ago (~45d)
    sir_zscore_1y         — SIR vs its own 1-year history (rolling z-score)
                            Identifies when shorting is abnormally elevated
                            for that specific stock, regardless of the absolute level

    All features are ranked cross-sectionally in load_and_prepare_data.

    @param si_raw: Raw short-interest dataframe.
    @param price_df: Daily price dataframe used for date alignment.
    @return: Tuple of merged short-interest feature dataframe and feature names.
    """
    si = (
        si_raw
        .rename(columns={"short_interest": "shares_short"})
        [["symbol", "settlement_date", "shares_short"]]
        .dropna(subset=["symbol", "shares_short"])
        .sort_values(["symbol", "settlement_date"])
    )

    # Compute 1-period and 3-period pct changes within each stock
    si["si_chg_1period"] = si.groupby("symbol")["shares_short"].pct_change(1)
    si["si_chg_3period"] = si.groupby("symbol")["shares_short"].pct_change(3)

    # Forward-fill settlement data onto daily price rows via merge_asof
    price_dates = (
        price_df[["symbol", "date", "avg_volume_20d"]]
        .drop_duplicates(subset=["symbol", "date"])
        .sort_values("date")
    )

    si_daily = pd.merge_asof(
        price_dates,
        si.rename(columns={"settlement_date": "date"}).sort_values("date"),
        on="date",
        by="symbol",
        direction="backward",   # only use reports available on or before each date
    )

    # Days-to-cover
    si_daily["short_interest_ratio"] = (
        si_daily["shares_short"] / si_daily["avg_volume_20d"].clip(lower=1)
    )

    # Rolling 1-year z-score of SIR (24 bi-monthly periods ≈ 1 year)
    def rolling_zscore(s, window=24):
        """
        Compute a rolling z-score for one time series.

        @param s: Input series to normalize.
        @param window: Rolling lookback window length.
        @return: Rolling z-score series.
        """
        m   = s.rolling(window, min_periods=6).mean()
        std = s.rolling(window, min_periods=6).std().clip(lower=1e-8)
        return (s - m) / std

    si_daily["sir_zscore_1y"] = (
        si_daily
        .sort_values(["symbol", "date"])
        .groupby("symbol")["short_interest_ratio"]
        .transform(rolling_zscore)
    )

    out_cols = ["symbol", "date"] + SI_FEATURE_COLS
    available = [c for c in out_cols if c in si_daily.columns]
    return si_daily[available], [c for c in SI_FEATURE_COLS if c in si_daily.columns]


# ══════════════════════════════════════════════════════════════════════════════
# Feature engineering — fundamentals
# ══════════════════════════════════════════════════════════════════════════════

def build_fundamental_features(
    fund_raw: pd.DataFrame,
    price_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list]:
    """
    Join quarterly fundamental features to daily price rows using merge_asof
    on `report_available_date` (= quarter end + 45-day reporting lag).

    The 45-day lag ensures no lookahead: for a row on date T, we only use
    reports whose `report_available_date` ≤ T, meaning earnings that were
    already public by that date.

    @param fund_raw: Raw quarterly fundamentals dataframe.
    @param price_df: Daily price dataframe used for date alignment.
    @return: Tuple of merged fundamentals dataframe and feature names.
    """
    fund = fund_raw.copy()
    fund["report_available_date"] = pd.to_datetime(fund["report_available_date"])
    fund = fund.sort_values(["symbol", "report_available_date"])

    feat_cols = [c for c in FUNDAMENTAL_FEATURE_COLS if c in fund.columns]
    if not feat_cols:
        return pd.DataFrame(columns=["symbol", "date"]), []

    price_dates = (
        price_df[["symbol", "date"]]
        .drop_duplicates()
        .sort_values("date")
    )

    fund_daily = pd.merge_asof(
        price_dates,
        fund[["symbol", "report_available_date"] + feat_cols]
            .rename(columns={"report_available_date": "date"})
            .sort_values("date"),
        on="date",
        by="symbol",
        direction="backward",
    )

    return fund_daily[["symbol", "date"] + feat_cols], feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def print_merge_coverage(df: pd.DataFrame, feature_cols: list[str], label: str) -> None:
    """
    Print merge coverage overall and by country for a feature block.

    A row is counted as "covered" if at least one column in `feature_cols`
    is non-null after the merge.

    @param df: Dataframe after the feature merge.
    @param feature_cols: Feature columns to check for coverage.
    @param label: Human-readable label for the feature block.
    @return: None.
    """
    if not feature_cols:
        print(f"    [coverage] {label}: no feature columns found")
        return

    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        print(f"    [coverage] {label}: columns missing after merge")
        return

    has_data = df[cols].notna().any(axis=1)
    print(f"    [coverage] {label} (>=1 non-null): {has_data.mean():.2%}")

    by_country = (
        pd.DataFrame({"country": df["country"], "has_data": has_data})
        .groupby("country", dropna=False)["has_data"]
        .mean()
        .sort_index()
    )
    for country, rate in by_country.items():
        vprint(f"      {country}: {rate:.2%}")


def load_and_prepare_data() -> tuple[pd.DataFrame, list]:
    """
    Load the base CSV, optionally merge short interest + fundamentals, add
    regime flags and cross-sectional ranks.

    Short interest and fundamentals are loaded only if their data files exist
    (created by fetch_short_interest.py / fetch_fundamentals.py).  If a file
    is missing the corresponding features are simply omitted and a warning is
    printed, so the pipeline can run without them during initial development.

    Returns (enriched_df, all_numeric_features).
    All feature engineering runs on the full df before any split so that
    cross-sectional ranks are computed consistently across the entire universe.

    @return: Tuple of prepared dataframe and numeric feature list.
    """
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    extra_numeric: list[str] = []

    # ── short interest features ────────────────────────────────────────────────
    if USE_SHORT_INTEREST:
        if SI_PATH.exists():
            print("  Loading short interest data...")
            si_raw  = pd.read_parquet(SI_PATH)
            si_df, si_cols = build_short_interest_features(si_raw, df)
            df = df.merge(si_df, on=["symbol", "date"], how="left")
            extra_numeric.extend(si_cols)
            print_merge_coverage(df, si_cols, label="short interest")
            FEATURES_TO_RANK.extend(
                [c for c in ["short_interest_ratio", "sir_zscore_1y"]
                 if c in si_cols and c not in FEATURES_TO_RANK]
            )
            vprint(f"    Added {len(si_cols)} SI features: {si_cols}")
        else:
            print(f"  [skip] Short interest file not found: {SI_PATH}")
            print("         Run fetch_short_interest.py to generate it.")

    # ── fundamental features ───────────────────────────────────────────────────
    if USE_FUNDAMENTALS:
        if FUNDAMENTALS_PATH.exists():
            print("  Loading fundamental data...")
            fund_raw = pd.read_parquet(FUNDAMENTALS_PATH)
            fund_df, fund_cols = build_fundamental_features(fund_raw, df)
            df = df.merge(fund_df, on=["symbol", "date"], how="left")
            extra_numeric.extend(fund_cols)
            print_merge_coverage(df, fund_cols, label="fundamentals")
            vprint(f"    Added {len(fund_cols)} fundamental features: {fund_cols}")
        else:
            print(f"  [skip] Fundamentals file not found: {FUNDAMENTALS_PATH}")
            print("         Run fetch_fundamentals.py to generate it.")

    # ── regime flags ──────────────────────────────────────────────────────────
    df, regime_cols = add_regime_features(df)

    # ── cross-sectional ranks ─────────────────────────────────────────────────
    rank_cols: list[str] = []
    if USE_RANK_FEATURES:
        df, rank_cols = add_cross_sectional_ranks(df, FEATURES_TO_RANK)

    all_numeric = BASE_NUMERIC_FEATURES + extra_numeric + regime_cols + rank_cols

    # Validate
    required = set(all_numeric + CATEGORICAL_FEATURES + [TARGET_CLF, TARGET_REG, DATE_COL, "symbol"])
    missing  = required - set(df.columns)
    # TARGET_REG might not exist if the CSV predates that column name; warn but continue
    if TARGET_REG not in df.columns:
        print(f"  [warn] '{TARGET_REG}' not in data — regression models will be skipped.")
    missing = missing - {TARGET_REG}
    if missing:
        raise ValueError(f"Missing required columns after feature engineering: {missing}")

    return df, all_numeric


# ══════════════════════════════════════════════════════════════════════════════
# Embargo + purging
# ══════════════════════════════════════════════════════════════════════════════

def apply_train_embargo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the last EMBARGO_CALENDAR_DAYS of training data.

    Rows near the train/val boundary have labels computed from early 2022 prices
    (the start of the rate-hike bear market), which would leak the val-period
    distribution into training.  Removing them eliminates this boundary leakage.

    @param df: Training dataframe before embargoing.
    @return: Embargoed training dataframe.
    """
    cutoff = pd.Timestamp(TRAIN_END) - pd.Timedelta(days=EMBARGO_CALENDAR_DAYS)
    return df[df[DATE_COL] <= cutoff].copy()


def purge_overlapping_labels(df: pd.DataFrame, stride: int = PURGE_STRIDE) -> pd.DataFrame:
    """
    Sub-sample training rows to reduce overlapping-label serial correlation.

    Consecutive daily rows share 59 out of 60 label-window days, making them
    nearly identical targets.  Keeping only 1 row per `stride` trading days per
    stock dramatically reduces this without discarding the full history.

    @param df: Training dataframe to purge.
    @param stride: Trading-day spacing used when sub-sampling each symbol.
    @return: Purged training dataframe.
    """
    df = df.copy().sort_values([DATE_COL, "symbol"])
    df["_cc"] = df.groupby("symbol").cumcount()
    purged = df[df["_cc"] % stride == 0].drop(columns=["_cc"])
    return purged.reset_index(drop=True)


def split_data(df: pd.DataFrame):
    """
    Produce the final train / validation / test DataFrames.

    Train:      up to TRAIN_END, with embargo + purge applied.
    Validation: VAL_START – VAL_END  (2022–2023, rate-hike + soft landing).
    Test:       TEST_START onward     (2024–2025, held-out, never used for selection).

    @param df: Fully prepared modeling dataframe.
    @return: Tuple of train, validation, and test dataframes.
    """
    train_df = df[df[DATE_COL] <= pd.Timestamp(TRAIN_END)].copy()
    val_df   = df[(df[DATE_COL] >= pd.Timestamp(VAL_START)) &
                  (df[DATE_COL] <= pd.Timestamp(VAL_END))].copy()
    test_df  = df[df[DATE_COL] >= pd.Timestamp(TEST_START)].copy()

    train_df = apply_train_embargo(train_df)
    train_df = purge_overlapping_labels(train_df)

    for name, s in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        if len(s) == 0:
            raise ValueError(f"{name} split is empty after preprocessing.")
    return train_df, val_df, test_df


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward fold generator
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_splits(
    df: pd.DataFrame,
    initial_train_years: int = 3,
    val_years: int = 1,
    step_years: int = 1,
) -> list:
    """
    Generate expanding-window (train, val, label) fold tuples for walk-forward CV.

    Each fold grows the training set by `step_years` and validates on the
    following `val_years`.  Embargo and purge are applied to each train fold so
    the CV reproduces the exact data conditions of the final model.

    Returns a list of (train_df, val_df, fold_label) tuples.

    @param df: Fully prepared modeling dataframe.
    @param initial_train_years: Initial training window length in years.
    @param val_years: Validation window length in years.
    @param step_years: Window expansion step in years.
    @return: List of walk-forward fold tuples.
    """
    df = df.sort_values(DATE_COL)
    min_date     = df[DATE_COL].min()
    folds        = []
    train_end_dt = min_date + pd.DateOffset(years=initial_train_years)

    while True:
        val_end_dt = train_end_dt + pd.DateOffset(years=val_years)
        if val_end_dt > df[DATE_COL].max():
            break

        eff_train_end = train_end_dt - pd.Timedelta(days=EMBARGO_CALENDAR_DAYS)
        train_fold    = df[df[DATE_COL] <= eff_train_end].copy()
        val_fold      = df[(df[DATE_COL] > train_end_dt) &
                           (df[DATE_COL] <= val_end_dt)].copy()

        if len(train_fold) > 0 and len(val_fold) > 0:
            train_fold = purge_overlapping_labels(train_fold, stride=PURGE_STRIDE)
            label = (f"val_{train_end_dt.year}–{int(val_end_dt.year)-1}"
                     if val_end_dt.year > train_end_dt.year
                     else f"val_{train_end_dt.year}")
            folds.append((train_fold, val_fold, label))

        train_end_dt += pd.DateOffset(years=step_years)

    return folds


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessors
# ══════════════════════════════════════════════════════════════════════════════

def make_standard_preprocessor(numeric_features: list, scale_numeric: bool) -> ColumnTransformer:
    """
    Build the ColumnTransformer used by LR, RF, Ridge, and RF-Regressor.

    Numeric block:  median imputation (keep_empty_features=True so all-NaN
                    columns like early-period fundamentals don't shrink the
                    output shape and break HGB's categorical_features index).
    Categorical:    most-frequent imputation + one-hot encoding.
    regime_vol:     single column passed through RegimeVolTransformer, which
                    fits a 70th-percentile threshold on X_train only.

    @param numeric_features: Numeric feature column names.
    @param scale_numeric: Whether to standard-scale numeric features.
    @return: Configured sklearn column transformer.
    """
    num_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
        # regime_vol_elevated: fitted on X_train only inside the Pipeline —
        # prevents vol-quantile threshold from leaking val/test distribution.
        ("regime_vol", Pipeline([
            ("imputer",  SimpleImputer(strategy="median")),
            ("vol_flag", RegimeVolTransformer(quantile=0.70)),
        ]), ["mkt_volatility_60d"]),
    ])


def make_histgb_preprocessor(numeric_features: list) -> ColumnTransformer:
    """
    Build the ColumnTransformer used by HistGradientBoosting models.

    Same structure as make_standard_preprocessor but uses OrdinalEncoder
    instead of OneHotEncoder for the categorical block.  HGB natively handles
    ordinal-encoded categoricals via its categorical_features parameter, which
    allows it to split on category boundaries rather than treating each level
    as an independent binary feature.

    @param numeric_features: Numeric feature column names.
    @return: Configured sklearn column transformer for HGB models.
    """
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
                encoded_missing_value=np.nan,
            )),
        ]), CATEGORICAL_FEATURES),
        ("regime_vol", Pipeline([
            ("imputer",  SimpleImputer(strategy="median")),
            ("vol_flag", RegimeVolTransformer(quantile=0.70)),
        ]), ["mkt_volatility_60d"]),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# Ensemble wrapper
# ══════════════════════════════════════════════════════════════════════════════

class AveragingEnsemble:
    """
    Simple probability-averaging ensemble over a list of fitted classifiers.
    Implements predict / predict_proba so it is a drop-in for any code that
    calls model.predict_proba(X)[:, 1].
    """
    def __init__(self, models: list):
        """
        Initialize the averaging ensemble with fitted classifiers.

        @param models: List of fitted classifier-like models.
        @return: None.
        """
        self.models = models

    def predict_proba(self, X):
        """
        Average positive-class probabilities across member models.

        @param X: Feature matrix to score.
        @return: Two-column class-probability array.
        """
        avg = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return np.column_stack([1 - avg, avg])

    def predict(self, X):
        """
        Convert averaged probabilities into hard class predictions.

        @param X: Feature matrix to score.
        @return: Integer class predictions.
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# Model registry — config-driven pipeline factory
# ══════════════════════════════════════════════════════════════════════════════
#
# Instead of one build_* function per model, each model is described by a
# config dict.  build_pipeline() turns any config into a full sklearn Pipeline.
#
# Config keys:
#   name           — identifier used in output filenames and print statements
#   prep           — preprocessor type: "standard" (LR / RF / Ridge)
#                    or "histgb" (HistGradientBoosting)
#   scale          — whether to StandardScale numeric features (True for LR/Ridge)
#   make_estimator — callable(nf) → unfitted sklearn estimator


def _make_hgb_estimator(cls, nf: list):
    """
    Instantiate a HistGradientBoosting estimator with the correct
    categorical_features index array.

    ColumnTransformer output order is [num | cat | regime_vol], so categorical
    columns occupy positions len(nf) … len(nf)+len(CATEGORICAL_FEATURES)-1.

    early_stopping is disabled because sklearn's internal random validation
    split has overlapping-label serial correlation, which causes spurious early
    stops on financial time-series data.

    @param cls: HistGradientBoosting estimator class to instantiate.
    @param nf: Numeric feature names used to locate categorical columns.
    @return: Configured HistGradientBoosting estimator instance.
    """
    # Categorical columns sit immediately after the numeric block in the
    # ColumnTransformer output — compute their integer positions here.
    cat_idx = list(range(len(nf), len(nf) + len(CATEGORICAL_FEATURES)))
    return cls(
        learning_rate=HGB_LEARNING_RATE, max_iter=HGB_MAX_ITER,
        max_depth=HGB_MAX_DEPTH, min_samples_leaf=HGB_MIN_SAMPLES_LEAF,
        l2_regularization=HGB_L2,
        early_stopping=False,
        categorical_features=cat_idx,
        random_state=RANDOM_STATE,
    )


# Classification model registry
CLF_CONFIGS = [
    {
        "name":           "dummy",
        "prep":           "standard",
        "scale":          False,
        "make_estimator": lambda _: DummyClassifier(strategy="prior"),
    },
    {
        "name":  "logistic_regression",
        "prep":  "standard",
        "scale": True,   # LR is sensitive to feature scale
        "make_estimator": lambda _: LogisticRegression(
            C=LR_C, max_iter=2000, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    },
    {
        "name":  "random_forest",
        "prep":  "standard",
        "scale": False,
        "make_estimator": lambda _: RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            max_features=RF_MAX_FEATURES,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    },
    {
        "name":           "hist_gradient_boosting",
        "prep":           "histgb",
        "scale":          False,
        "make_estimator": lambda nf: _make_hgb_estimator(HistGradientBoostingClassifier, nf),
    },
]

# Regression model registry
REG_CONFIGS = [
    {
        "name":           "ridge",
        "prep":           "standard",
        "scale":          True,
        "make_estimator": lambda _: Ridge(alpha=RIDGE_ALPHA),
    },
    {
        "name":  "rf_regressor",
        "prep":  "standard",
        "scale": False,
        "make_estimator": lambda _: RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            max_features=RF_MAX_FEATURES,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    },
    {
        "name":           "hgb_regressor",
        "prep":           "histgb",
        "scale":          False,
        "make_estimator": lambda nf: _make_hgb_estimator(HistGradientBoostingRegressor, nf),
    },
]


def build_pipeline(config: dict, nf: list) -> Pipeline:
    """
    Assemble a two-step sklearn Pipeline from a model config entry.

    @param config: One entry from `CLF_CONFIGS` or `REG_CONFIGS`.
    @param nf: Numeric feature column names.
    @return: Unfitted pipeline with preprocessing and model steps.
    """
    if config["prep"] == "histgb":
        prep = make_histgb_preprocessor(nf)
    else:
        prep = make_standard_preprocessor(nf, scale_numeric=config["scale"])
    return Pipeline([("preprocessor", prep), ("model", config["make_estimator"](nf))])


def get_clf_models(nf: list) -> dict:
    """
    Build fresh classifier pipelines for all configured classification models.

    @param nf: Numeric feature column names.
    @return: Mapping of classifier names to unfitted pipelines.
    """
    return {c["name"]: build_pipeline(c, nf) for c in CLF_CONFIGS}


def get_reg_models(nf: list) -> dict:
    """
    Build fresh regression pipelines for all configured regression models.

    @param nf: Numeric feature column names.
    @return: Mapping of regressor names to unfitted pipelines.
    """
    return {c["name"]: build_pipeline(c, nf) for c in REG_CONFIGS}


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def safe_auc(y_true, y_score) -> float:
    """
    Return ROC AUC, or NaN if only one class is present.

    @param y_true: True binary labels.
    @param y_score: Model scores for the positive class.
    @return: ROC AUC value or NaN.
    """
    return roc_auc_score(y_true, y_score) if len(np.unique(y_true)) >= 2 else np.nan


def evaluate_predictions(y_true, y_pred, y_score) -> dict:
    """
    Compute a standard set of binary classification metrics.

    @param y_true: True binary labels.
    @param y_pred: Predicted binary labels.
    @param y_score: Model scores for the positive class.
    @return: Dictionary of evaluation metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   safe_auc(y_true, y_score),
        "pr_auc":    average_precision_score(y_true, y_score),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def score_clf(model, X, y) -> tuple:
    """
    Score a classifier and return (metrics_dict, y_pred, y_score).

    y_score is the positive-class probability if predict_proba is available,
    otherwise the hard prediction cast to float.

    @param model: Fitted classifier-like model.
    @param X: Feature matrix to score.
    @param y: True binary labels.
    @return: Tuple of metrics dictionary, predictions, and scores.
    """
    y_pred  = model.predict(X)
    y_score = (model.predict_proba(X)[:, 1]
               if hasattr(model, "predict_proba") else y_pred.astype(float))
    return evaluate_predictions(y, y_pred, y_score), y_pred, y_score


def score_reg(model, X, y_reg, y_clf) -> tuple:
    """
    Evaluate a regression model's RANKING quality using the binary label.

    The model is trained to predict future_drawdown_60d (continuous, negative
    values = drawdown).  Risk score = –predicted_drawdown, so stocks predicted
    to fall most are ranked highest.  We evaluate that ranking against the
    binary drawdown label using the same AUC / lift metrics as classifiers.

    Returns (metrics_dict, risk_scores_array).

    @param model: Fitted regression model.
    @param X: Feature matrix to score.
    @param y_reg: Continuous regression target values.
    @param y_clf: Binary classification labels used for ranking metrics.
    @return: Tuple of metrics dictionary and derived risk scores.
    """
    pred_drawdown = model.predict(X)
    risk_score    = -pred_drawdown   # flip: more negative prediction = higher rank

    roc  = safe_auc(y_clf, risk_score)
    pr   = average_precision_score(y_clf, risk_score)
    mae  = mean_absolute_error(y_reg, pred_drawdown)
    r2   = r2_score(y_reg, pred_drawdown)
    lift = top_k_event_rate(y_clf, risk_score, k_frac=0.10)

    metrics = {
        "roc_auc": roc, "pr_auc": pr,
        "mae": mae, "r2": r2,
        "top10_lift": lift["lift"],
    }
    return metrics, risk_score


def evaluate_at_threshold(y_true, y_score, threshold) -> dict:
    """
    Binarize model scores at a threshold and compute classification metrics.

    @param y_true: True binary labels.
    @param y_score: Model scores for the positive class.
    @param threshold: Decision threshold applied to `y_score`.
    @return: Dictionary of evaluation metrics at the threshold.
    """
    return evaluate_predictions(y_true, (y_score >= threshold).astype(int), y_score)


def threshold_sweep(y_true, y_score, model_name, split_name, thresholds=None) -> pd.DataFrame:
    """
    Evaluate a score vector at multiple decision thresholds and return a DataFrame.

    Useful for selecting an operating point when a hard yes/no label is required
    (e.g. generating alerts).  Not needed when using the model purely for ranking.

    @param y_true: True binary labels.
    @param y_score: Model scores for the positive class.
    @param model_name: Model identifier to store in the output.
    @param split_name: Dataset split name to store in the output.
    @param thresholds: Optional iterable of thresholds to evaluate.
    @return: Dataframe of per-threshold metrics.
    """
    thresholds = thresholds or np.arange(0.05, 0.55, 0.05)
    rows = []
    for t in thresholds:
        m = evaluate_at_threshold(y_true, y_score, t)
        m.update({"model": model_name, "split": split_name, "threshold": float(t)})
        rows.append(m)
    return pd.DataFrame(rows)


def top_k_event_rate(y_true, y_score, k_frac=0.10) -> dict:
    """
    Measure how enriched for true positives the top-k% of scored rows are.

    This is the primary business metric: if we flag the top decile of stocks
    by risk score, what fraction of them actually draw down 20%+?  The lift
    (= top-k event rate / base rate) tells us how much better than random we are.

    @param y_true: True binary labels.
    @param y_score: Model scores used for ranking.
    @param k_frac: Fraction of highest-scored rows to inspect.
    @return: Dictionary summarizing top-k enrichment metrics.
    """
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    k       = max(1, int(np.ceil(len(y_true) * k_frac)))
    top_idx = np.argsort(-y_score)[:k]
    top_rate  = float(np.mean(y_true[top_idx]))
    base_rate = float(np.mean(y_true))
    return {
        "k_frac": float(k_frac), "top_k_n": int(k),
        "top_k_event_rate": top_rate, "base_rate": base_rate,
        "lift": float(top_rate / base_rate) if base_rate > 0 else np.nan,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_predictions(df_split, model_name, split_name, y_true, y_pred, y_score):
    """
    Save per-row predictions for a dataset split to CSV.

    @param df_split: Source dataframe for the split being saved.
    @param model_name: Model identifier used in the output filename.
    @param split_name: Dataset split name used in the output filename.
    @param y_true: True labels.
    @param y_pred: Predicted labels or `None` when not applicable.
    @param y_score: Model scores saved alongside the labels.
    @return: None.
    """
    out = df_split[["date", "symbol", "country", "sector"]].copy()
    out["y_true"]  = np.asarray(y_true)
    out["y_pred"]  = np.asarray(y_pred) if y_pred is not None else np.nan
    out["y_score"] = np.asarray(y_score)
    out.to_csv(TABLES_DIR / f"{model_name}_{split_name}_predictions.csv", index=False)


def save_logistic_coefficients(pipeline):
    """
    Save logistic-regression coefficients sorted by absolute magnitude.

    @param pipeline: Fitted logistic-regression pipeline.
    @return: None.
    """
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    pd.DataFrame({
        "feature":         feature_names,
        "coefficient":     coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False).to_csv(
        TABLES_DIR / "logistic_regression_coefficients.csv", index=False
    )


def save_rf_importances(pipeline, filename: str = "random_forest_feature_importances.csv"):
    """
    Save random-forest feature importances sorted in descending order.

    @param pipeline: Fitted random-forest pipeline.
    @param filename: Output CSV filename.
    @return: None.
    """
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances   = pipeline.named_steps["model"].feature_importances_
    pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).to_csv(TABLES_DIR / filename, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward cross-validation
# ══════════════════════════════════════════════════════════════════════════════

def run_walk_forward_cv(df: pd.DataFrame, numeric_features: list) -> pd.DataFrame:
    """
    Evaluate both classification and regression models across 4 expanding
    temporal folds, each covering a distinct market regime.  The fold-average
    metrics are a much more honest estimate of generalisation than the single
    final val split.

    @param df: Fully prepared modeling dataframe.
    @param numeric_features: Numeric feature column names.
    @return: Walk-forward cross-validation metrics dataframe.
    """
    print("\nWalk-forward CV...")

    folds = walk_forward_splits(df)
    if not folds:
        print("  Not enough data for walk-forward CV.")
        return pd.DataFrame()

    clf_names = ["logistic_regression", "random_forest", "hist_gradient_boosting"]
    reg_names = ["ridge", "rf_regressor", "hgb_regressor"]
    has_reg   = TARGET_REG in df.columns and RUN_REGRESSION_MODELS

    all_rows = []

    for fold_idx, (train_fold, val_fold, fold_label) in enumerate(folds, 1):
        pos_rate = val_fold[TARGET_CLF].mean()
        vprint(
            f"\nFold {fold_idx}: {fold_label}  "
            f"(train={len(train_fold):,}, val={len(val_fold):,}, "
            f"pos_rate={pos_rate:.4f})"
        )

        X_tr   = train_fold[numeric_features + CATEGORICAL_FEATURES]
        y_tr   = train_fold[TARGET_CLF].astype(int)
        X_vl   = val_fold[numeric_features + CATEGORICAL_FEATURES]
        y_vl   = val_fold[TARGET_CLF].astype(int)

        # ── classification ─────────────────────────────────────────────────────
        fold_clfs: dict = {}
        for name in clf_names:
            pipe = get_clf_models(numeric_features)[name]
            pipe.fit(X_tr, y_tr)
            fold_clfs[name] = pipe
            metrics, _, y_score = score_clf(pipe, X_vl, y_vl)
            lift = top_k_event_rate(y_vl, y_score, k_frac=0.10)
            vprint(
                f"  [clf] {name:25s}  ROC={metrics['roc_auc']:.4f}  "
                f"PR={metrics['pr_auc']:.4f}  lift@10%={lift['lift']:.2f}x"
            )
            all_rows.append({
                "fold": fold_label, "model": name, "model_type": "classifier",
                "val_roc_auc": metrics["roc_auc"],
                "val_pr_auc":  metrics["pr_auc"],
                "top10_lift":  lift["lift"],
            })

        # ── ensemble RF + LR ───────────────────────────────────────────────────
        if "random_forest" in fold_clfs and "logistic_regression" in fold_clfs:
            ens = AveragingEnsemble([fold_clfs["random_forest"],
                                     fold_clfs["logistic_regression"]])
            metrics, _, ens_score = score_clf(ens, X_vl, y_vl)
            lift = top_k_event_rate(y_vl, ens_score, k_frac=0.10)
            vprint(
                f"  [clf] {'ensemble_rf_lr':25s}  ROC={metrics['roc_auc']:.4f}  "
                f"PR={metrics['pr_auc']:.4f}  lift@10%={lift['lift']:.2f}x"
            )
            all_rows.append({
                "fold": fold_label, "model": "ensemble_rf_lr",
                "model_type": "classifier",
                "val_roc_auc": metrics["roc_auc"],
                "val_pr_auc":  metrics["pr_auc"],
                "top10_lift":  lift["lift"],
            })

        # ── regression ─────────────────────────────────────────────────────────
        if has_reg:
            y_tr_reg = train_fold[TARGET_REG].astype(float)
            y_vl_reg = val_fold[TARGET_REG].astype(float)

            for name in reg_names:
                pipe = get_reg_models(numeric_features)[name]
                pipe.fit(X_tr, y_tr_reg)
                metrics, _ = score_reg(pipe, X_vl, y_vl_reg, y_vl)
                vprint(
                    f"  [reg] {name:25s}  ROC={metrics['roc_auc']:.4f}  "
                    f"PR={metrics['pr_auc']:.4f}  "
                    f"lift@10%={metrics['top10_lift']:.2f}x  "
                    f"MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}"
                )
                all_rows.append({
                    "fold": fold_label, "model": name, "model_type": "regressor",
                    "val_roc_auc": metrics["roc_auc"],
                    "val_pr_auc":  metrics["pr_auc"],
                    "top10_lift":  metrics["top10_lift"],
                    "mae":         metrics["mae"],
                    "r2":          metrics["r2"],
                })

    cv_df = pd.DataFrame(all_rows)

    avg = (
        cv_df.groupby(["model_type", "model"])[["val_roc_auc", "val_pr_auc", "top10_lift"]]
        .mean()
        .sort_values("val_roc_auc", ascending=False)
    )
    if VERBOSE:
        print("\nWalk-forward CV — averages across folds (all models):")
        print(avg.round(4).to_string())
    else:
        clf_avg = avg.loc["classifier"] if "classifier" in avg.index.get_level_values(0) else pd.DataFrame()
        reg_avg = avg.loc["regressor"] if "regressor" in avg.index.get_level_values(0) else pd.DataFrame()
        if len(clf_avg):
            best_clf = clf_avg.sort_values("val_pr_auc", ascending=False).head(1)
            name = best_clf.index[0]
            row = best_clf.iloc[0]
            print(
                f"  CV best classifier: {name} | "
                f"ROC={row['val_roc_auc']:.4f} PR={row['val_pr_auc']:.4f} "
                f"lift@10%={row['top10_lift']:.2f}x"
            )
        if len(reg_avg):
            best_reg = reg_avg.sort_values("val_roc_auc", ascending=False).head(1)
            name = best_reg.index[0]
            row = best_reg.iloc[0]
            print(
                f"  CV best regressor:  {name} | "
                f"ROC={row['val_roc_auc']:.4f} PR={row['val_pr_auc']:.4f} "
                f"lift@10%={row['top10_lift']:.2f}x"
            )

    cv_df.to_csv(TABLES_DIR / "walk_forward_cv_results.csv", index=False)
    print(f"  Saved walk-forward CV results → {TABLES_DIR / 'walk_forward_cv_results.csv'}")

    return cv_df


# ══════════════════════════════════════════════════════════════════════════════
# Final split training + evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_final_split(df: pd.DataFrame, numeric_features: list, has_reg: bool) -> None:
    """
    Train and evaluate all models on the definitive train / val / test split.

    Covers three steps:
      1. Classification — Dummy, LR, RF, HGB, then RF+LR ensemble.
      2. Regression     — Ridge, RF-Reg, HGB-Reg (skipped if has_reg=False).
    3. Reporting      — picks the best model on val, saves metrics / predictions /
                          feature importances / threshold sweep / lift CSVs to OUT_DIR.

    Nothing in this function is used to make modelling decisions — model
    selection is based on walk-forward CV.  This split gives the final
    held-out numbers reported in the README.

    @param df: Fully prepared modeling dataframe.
    @param numeric_features: Numeric feature column names.
    @param has_reg: Whether regression targets and models are available.
    @return: None.
    """
    train_df, val_df, test_df = split_data(df)

    # Build feature matrices for all three splits
    feat_cols = numeric_features + CATEGORICAL_FEATURES
    X_train, y_train = train_df[feat_cols], train_df[TARGET_CLF].astype(int)
    X_val,   y_val   = val_df[feat_cols],   val_df[TARGET_CLF].astype(int)
    X_test,  y_test  = test_df[feat_cols],  test_df[TARGET_CLF].astype(int)

    y_train_reg = train_df[TARGET_REG].astype(float) if has_reg else None
    y_val_reg   = val_df[TARGET_REG].astype(float)   if has_reg else None
    y_test_reg  = test_df[TARGET_REG].astype(float)  if has_reg else None

    all_metrics: list[dict] = []

    # ── classification ─────────────────────────────────────────────────────────
    print("\nFinal split evaluation...")
    print(f"  rows: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    if VERBOSE:
        print("  CLASSIFIERS")
        print(f"  {'model':<26}  {'val_ROC':>7}  {'val_PR':>6}  {'test_ROC':>8}  {'test_PR':>7}")
        print("  " + "-" * 58)

    fitted_clfs: dict = {}

    for model_name, pipe in get_clf_models(numeric_features).items():
        pipe.fit(X_train, y_train)
        fitted_clfs[model_name] = pipe

        # Score on all three splits; cache val/test to avoid redundant inference
        val_m,  val_pred,  val_score  = score_clf(pipe, X_val,  y_val)
        test_m, test_pred, test_score = score_clf(pipe, X_test, y_test)

        for split_name, X_s, y_s, df_s, m, pred, score in [
            ("train",      X_train, y_train, train_df, *score_clf(pipe, X_train, y_train)),
            ("validation", X_val,   y_val,   val_df,   val_m,  val_pred,  val_score),
            ("test",       X_test,  y_test,  test_df,  test_m, test_pred, test_score),
        ]:
            m.update({"model": model_name, "split": split_name, "model_type": "classifier"})
            all_metrics.append(m)
            save_predictions(df_s, model_name, split_name, y_s, pred, score)

        vprint(
            f"  {model_name:<26}  {val_m['roc_auc']:>7.4f}  {val_m['pr_auc']:>6.4f}"
            f"  {test_m['roc_auc']:>8.4f}  {test_m['pr_auc']:>7.4f}"
        )

    # ── ensemble: average RF + LR probability scores ───────────────────────────
    # RF and LR have complementary failure modes (RF struggles in low-vol bull
    # markets; LR struggles in high-vol bear markets), so averaging their scores
    # smooths out regime-specific weaknesses.
    if "random_forest" in fitted_clfs and "logistic_regression" in fitted_clfs:
        ens = AveragingEnsemble([fitted_clfs["random_forest"],
                                  fitted_clfs["logistic_regression"]])
        fitted_clfs["ensemble_rf_lr"] = ens

        val_m,  val_pred,  val_score  = score_clf(ens, X_val,  y_val)
        test_m, test_pred, test_score = score_clf(ens, X_test, y_test)

        for split_name, X_s, y_s, df_s, m, pred, score in [
            ("train",      X_train, y_train, train_df, *score_clf(ens, X_train, y_train)),
            ("validation", X_val,   y_val,   val_df,   val_m,  val_pred,  val_score),
            ("test",       X_test,  y_test,  test_df,  test_m, test_pred, test_score),
        ]:
            m.update({"model": "ensemble_rf_lr", "split": split_name,
                      "model_type": "classifier"})
            all_metrics.append(m)
            save_predictions(df_s, "ensemble_rf_lr", split_name, y_s, pred, score)

        vprint(
            f"  {'ensemble_rf_lr':<26}  {val_m['roc_auc']:>7.4f}  {val_m['pr_auc']:>6.4f}"
            f"  {test_m['roc_auc']:>8.4f}  {test_m['pr_auc']:>7.4f}"
        )

    # ── regression ─────────────────────────────────────────────────────────────
    fitted_regs: dict = {}

    if has_reg:
        if VERBOSE:
            print("\n  REGRESSORS (ranked by -pred_drawdown)")
            print(f"  {'model':<26}  {'val_ROC':>7}  {'val_PR':>6}  {'test_ROC':>8}  {'test_PR':>7}")
            print("  " + "-" * 58)

        for model_name, pipe in get_reg_models(numeric_features).items():
            pipe.fit(X_train, y_train_reg)
            fitted_regs[model_name] = pipe

            for split_name, X_s, y_s_reg, y_s_clf, df_s in [
                ("train",      X_train, y_train_reg, y_train, train_df),
                ("validation", X_val,   y_val_reg,   y_val,   val_df),
                ("test",       X_test,  y_test_reg,  y_test,  test_df),
            ]:
                metrics, risk_score = score_reg(pipe, X_s, y_s_reg, y_s_clf)
                metrics.update({"model": model_name, "split": split_name,
                                 "model_type": "regressor"})
                all_metrics.append(metrics)
                save_predictions(df_s, model_name, split_name, y_s_clf, None, risk_score)

            # Look up the just-appended val/test rows for the print statement
            val_m  = next(m for m in reversed(all_metrics)
                          if m["model"] == model_name and m["split"] == "validation")
            test_m = next(m for m in reversed(all_metrics)
                          if m["model"] == model_name and m["split"] == "test")
            vprint(
                f"  {model_name:<26}  {val_m['roc_auc']:>7.4f}  {val_m['pr_auc']:>6.4f}"
                f"  {test_m['roc_auc']:>8.4f}  {test_m['pr_auc']:>7.4f}"
            )

    # ── model selection ────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(TABLES_DIR / "stage1_metrics.csv", index=False)

    # Best classifier by val PR AUC (more informative than ROC for imbalanced data)
    val_clf       = metrics_df[(metrics_df["split"] == "validation") &
                                (metrics_df["model_type"] == "classifier")]
    best_clf_name = val_clf.sort_values("pr_auc", ascending=False).iloc[0]["model"]

    # Best regressor by val ROC AUC (measures ranking quality)
    best_reg_name = None
    if has_reg:
        val_reg = metrics_df[(metrics_df["split"] == "validation") &
                              (metrics_df["model_type"] == "regressor")]
        if not val_reg.empty:
            best_reg_name = val_reg.sort_values("roc_auc", ascending=False).iloc[0]["model"]

    with open(OUT_DIR / "best_model.txt", "w") as f:
        f.write(str(best_clf_name))

    best_clf_val = val_clf.sort_values("pr_auc", ascending=False).iloc[0]
    best_clf_test = metrics_df[
        (metrics_df["split"] == "test")
        & (metrics_df["model_type"] == "classifier")
        & (metrics_df["model"] == best_clf_name)
    ].iloc[0]
    print(
        f"\nBest classifier: {best_clf_name} | "
        f"val PR={best_clf_val['pr_auc']:.4f}, val ROC={best_clf_val['roc_auc']:.4f} | "
        f"test PR={best_clf_test['pr_auc']:.4f}, test ROC={best_clf_test['roc_auc']:.4f}"
    )
    if best_reg_name:
        best_reg_val = metrics_df[
            (metrics_df["split"] == "validation")
            & (metrics_df["model_type"] == "regressor")
            & (metrics_df["model"] == best_reg_name)
        ].iloc[0]
        best_reg_test = metrics_df[
            (metrics_df["split"] == "test")
            & (metrics_df["model_type"] == "regressor")
            & (metrics_df["model"] == best_reg_name)
        ].iloc[0]
        print(
            f"Best regressor:  {best_reg_name} | "
            f"val ROC={best_reg_val['roc_auc']:.4f}, val PR={best_reg_val['pr_auc']:.4f} | "
            f"test ROC={best_reg_test['roc_auc']:.4f}, test PR={best_reg_test['pr_auc']:.4f}"
        )

    # ── feature importances ────────────────────────────────────────────────────
    if "logistic_regression" in fitted_clfs:
        save_logistic_coefficients(fitted_clfs["logistic_regression"])
    if "random_forest" in fitted_clfs:
        save_rf_importances(fitted_clfs["random_forest"], "random_forest_clf_importances.csv")
    if "rf_regressor" in fitted_regs:
        save_rf_importances(fitted_regs["rf_regressor"], "random_forest_reg_importances.csv")

    # ── threshold sweep + lift for best classifier ─────────────────────────────
    # Saves CSVs for offline analysis; only lift@10% is printed to console.
    best_clf = fitted_clfs.get(best_clf_name)
    if best_clf and hasattr(best_clf, "predict_proba"):
        val_score  = best_clf.predict_proba(X_val)[:, 1]
        test_score = best_clf.predict_proba(X_test)[:, 1]

        val_thr  = threshold_sweep(y_val,  val_score,  best_clf_name, "validation")
        test_thr = threshold_sweep(y_test, test_score, best_clf_name, "test")
        val_thr.to_csv(TABLES_DIR / f"{best_clf_name}_validation_threshold_sweep.csv", index=False)
        test_thr.to_csv(TABLES_DIR / f"{best_clf_name}_test_threshold_sweep.csv",       index=False)

        best_thr = float(val_thr.sort_values("f1", ascending=False).iloc[0]["threshold"])
        with open(OUT_DIR / f"{best_clf_name}_selected_threshold.txt", "w") as f:
            f.write(str(best_thr))

        pd.DataFrame([{
            "model": best_clf_name, "split": "test_selected_threshold",
            "threshold": best_thr,
            **evaluate_at_threshold(y_test, test_score, best_thr),
        }]).to_csv(TABLES_DIR / f"{best_clf_name}_test_selected_threshold_metrics.csv", index=False)

        val_lift  = top_k_event_rate(y_val,  val_score,  k_frac=0.10)
        test_lift = top_k_event_rate(y_test, test_score, k_frac=0.10)
        pd.DataFrame([{"split": "validation", **val_lift},
                      {"split": "test",       **test_lift}]
                     ).to_csv(TABLES_DIR / f"{best_clf_name}_top10pct_lift.csv", index=False)

        print(f"\n  {best_clf_name} lift@10%:  "
              f"val={val_lift['lift']:.2f}x  test={test_lift['lift']:.2f}x")

    # ── lift for best regressor ────────────────────────────────────────────────
    if best_reg_name and best_reg_name in fitted_regs:
        best_reg = fitted_regs[best_reg_name]
        _, val_risk  = score_reg(best_reg, X_val,  y_val_reg,  y_val)
        _, test_risk = score_reg(best_reg, X_test, y_test_reg, y_test)

        val_lift_reg  = top_k_event_rate(y_val,  val_risk,  k_frac=0.10)
        test_lift_reg = top_k_event_rate(y_test, test_risk, k_frac=0.10)
        pd.DataFrame([{"split": "validation", **val_lift_reg},
                      {"split": "test",       **test_lift_reg}]
                     ).to_csv(TABLES_DIR / f"{best_reg_name}_top10pct_lift.csv", index=False)

        print(f"  {best_reg_name} lift@10%:  "
              f"val={val_lift_reg['lift']:.2f}x  test={test_lift_reg['lift']:.2f}x")

    print(f"\nSaved all metrics and predictions → {OUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train stage-1 drawdown-risk models with time-aware evaluation."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full fold-by-fold and per-model evaluation details.",
    )
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = bool(args.verbose)

    # ── feature engineering ────────────────────────────────────────────────────
    print(f"Loading data and engineering features... (verbose={VERBOSE})")
    df, numeric_features = load_and_prepare_data()
    n_base  = len(BASE_NUMERIC_FEATURES)
    n_total = len(numeric_features)
    print(f"  Loaded {len(df):,} rows | {n_total} numeric features "
          f"({n_base} base + {n_total - n_base} engineered)")

    has_reg = TARGET_REG in df.columns and RUN_REGRESSION_MODELS

    # ── walk-forward CV ────────────────────────────────────────────────────────
    if RUN_WALK_FORWARD_CV:
        run_walk_forward_cv(df, numeric_features)

    # ── final train / val / test split ─────────────────────────────────────────
    run_final_split(df, numeric_features, has_reg)


if __name__ == "__main__":
    main()
