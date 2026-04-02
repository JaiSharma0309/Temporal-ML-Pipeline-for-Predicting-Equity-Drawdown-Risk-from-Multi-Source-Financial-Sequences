"""
Build the stage 1 modeling dataset for the equity drawdown risk project.

This script reads raw daily OHLCV price files downloaded from yfinance,
loads universe metadata with country/sector/benchmark mappings, engineers
rolling technical and benchmark-relative features, creates the future
20% drawdown-within-60-trading-days label, and saves both a full processed
dataset and a cleaned ML-ready dataset for downstream model training.
"""

from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

RAW_DIR = REPO_ROOT / "data/raw/prices_yfinance"
META_PATH = REPO_ROOT / "data/metadata/equity_universe_metadata.csv"
OUT_DIR = REPO_ROOT / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

US_BENCHMARK = "SPY"
CA_BENCHMARK = "XIU.TO"
MARKET_BENCHMARKS = {US_BENCHMARK, CA_BENCHMARK}

FUTURE_HORIZON = 60
DRAWDOWN_THRESHOLD = 0.20

def load_price_file(path: Path) -> pd.DataFrame:
    """
    Load a single raw price CSV and standardize its basic structure.

    The function checks that the expected OHLCV columns are present,
    parses the date column, sorts rows chronologically, and returns a
    clean per-symbol price dataframe ready for feature engineering.
    """
    df = pd.read_csv(path)

    expected = {"date", "symbol", "open", "high", "low", "close", "adjusted_close", "volume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def load_metadata() -> pd.DataFrame:
    """
    Load and validate the equity universe metadata file.

    This metadata provides the symbol-level mappings needed later in the
    pipeline, including country, sector, market benchmark, and sector
    benchmark. Duplicate symbols are removed before returning the table.
    """
    meta = pd.read_csv(META_PATH)

    required = {
        "symbol",
        "country",
        "sector",
        "market_benchmark",
        "sector_benchmark",
    }
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata file missing columns: {missing}")

    meta = meta.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    return meta

def trailing_max_drawdown(series: pd.Series, window: int) -> pd.Series:
    """
    Compute the worst trailing drawdown observed within a rolling window.

    For each date, this measures the most negative drawdown relative to
    the rolling maximum price over the specified window.
    """
    roll_max = series.rolling(window).max()
    drawdown = series / roll_max - 1.0
    return drawdown.rolling(window).min()

def downside_volatility(returns: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling downside volatility over a given window.

    Only negative returns are kept when estimating the rolling standard
    deviation, which makes this a downside-risk measure rather than a
    general volatility measure.
    """
    neg = returns.clip(upper=0)
    return neg.rolling(window).std()

def upside_volatility(returns: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling upside volatility over a given window.

    Only positive returns are kept when estimating the rolling standard
    deviation, which allows comparison between upside and downside risk.
    """
    pos = returns.clip(lower=0)
    return pos.rolling(window).std()

def days_since_last_high(series: pd.Series, window: int) -> pd.Series:
    """
    Measure how many days have passed since the most recent rolling high.

    Within each rolling window, this returns the number of periods since
    the last maximum value, which acts as a simple drawdown-duration feature.
    """
    return series.rolling(window).apply(lambda x: len(x) - 1 - np.argmax(x), raw=True)

def rolling_slope(log_price: pd.Series, window: int) -> pd.Series:
    """
    Estimate the rolling linear trend slope of log prices.

    The slope is computed within each rolling window using a centered time
    index, giving a compact measure of recent price trend direction and
    strength.
    """
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denom = np.sum(x_centered ** 2)

    def _slope(arr):
        if np.any(~np.isfinite(arr)):
            return np.nan
        return np.dot(x_centered, arr - arr.mean()) / denom

    return log_price.rolling(window).apply(_slope, raw=True)

def compute_future_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the forward-looking downside-risk label for supervised learning.

    For each date, the function looks ahead FUTURE_HORIZON trading days,
    finds the minimum future adjusted close, computes the forward drawdown,
    and labels the row as 1 if the stock experiences at least a
    DRAWDOWN_THRESHOLD decline over that future window.
    """
    px = df["adjusted_close"]

    future_min = pd.concat(
        [px.shift(-i) for i in range(1, FUTURE_HORIZON + 1)],
        axis=1
    ).min(axis=1)

    df["future_min_price_60d"] = future_min
    df["future_drawdown_60d"] = future_min / px - 1.0
    df["label_drawdown_20pct_60d"] = (future_min <= (1 - DRAWDOWN_THRESHOLD) * px).astype(int)

    last_valid_idx = len(df) - FUTURE_HORIZON
    if last_valid_idx < len(df):
        df.loc[last_valid_idx:, ["future_min_price_60d", "future_drawdown_60d", "label_drawdown_20pct_60d"]] = np.nan

    return df

def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer rolling price, volatility, drawdown, volume, and trend features.

    This function creates the core single-name feature set used in stage 1,
    including returns, volatility, downside/upside asymmetry, distance from
    highs, moving-average ratios, trend slopes, skewness, gap features,
    and intraday range features.
    """
    px = df["adjusted_close"]
    opn = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]
    vol = df["volume"]
    log_px = np.log(px)

    df["ret_1d"] = px.pct_change(1)
    df["ret_5d"] = px.pct_change(5)
    df["ret_20d"] = px.pct_change(20)
    df["ret_60d"] = px.pct_change(60)
    df["ret_120d"] = px.pct_change(120)

    df["volatility_20d"] = df["ret_1d"].rolling(20).std()
    df["volatility_60d"] = df["ret_1d"].rolling(60).std()
    df["downside_vol_60d"] = downside_volatility(df["ret_1d"], 60)
    df["upside_vol_60d"] = upside_volatility(df["ret_1d"], 60)
    df["downside_upside_vol_ratio_60d"] = df["downside_vol_60d"] / df["upside_vol_60d"]

    df["dist_from_60d_high"] = px / px.rolling(60).max() - 1.0
    df["dist_from_252d_high"] = px / px.rolling(252).max() - 1.0
    df["days_since_60d_high"] = days_since_last_high(px, 60)
    df["days_since_252d_high"] = days_since_last_high(px, 252)

    df["trailing_max_drawdown_60d"] = trailing_max_drawdown(px, 60)
    df["trailing_max_drawdown_252d"] = trailing_max_drawdown(px, 252)

    df["avg_volume_20d"] = vol.rolling(20).mean()
    df["avg_volume_60d"] = vol.rolling(60).mean()
    df["volume_spike_20d"] = vol / df["avg_volume_20d"]
    df["dollar_volume"] = px * vol
    df["avg_dollar_volume_20d"] = df["dollar_volume"].rolling(20).mean()

    df["ma_20d"] = px.rolling(20).mean()
    df["ma_60d"] = px.rolling(60).mean()
    df["ma_120d"] = px.rolling(120).mean()
    df["price_to_ma20"] = px / df["ma_20d"]
    df["price_to_ma60"] = px / df["ma_60d"]
    df["price_to_ma120"] = px / df["ma_120d"]
    df["ma20_to_ma60"] = df["ma_20d"] / df["ma_60d"]
    df["ma60_to_ma120"] = df["ma_60d"] / df["ma_120d"]

    df["log_price_slope_20d"] = rolling_slope(log_px, 20)
    df["log_price_slope_60d"] = rolling_slope(log_px, 60)
    df["log_price_slope_120d"] = rolling_slope(log_px, 120)
    df["trend_accel_20_60"] = df["log_price_slope_20d"] - df["log_price_slope_60d"]

    df["return_skew_20d"] = df["ret_1d"].rolling(20).skew()
    df["return_skew_60d"] = df["ret_1d"].rolling(60).skew()

    df["gap_open_prev_close"] = opn / close.shift(1) - 1.0
    df["intraday_range"] = (high - low) / close
    df["avg_intraday_range_20d"] = df["intraday_range"].rolling(20).mean()

    return df

def add_benchmark_features(df: pd.DataFrame, benchmark_df: pd.DataFrame, prefix: str, benchmark_symbol: str) -> pd.DataFrame:
    """
    Add benchmark and relative-performance features to an equity dataframe.

    The benchmark can be either the broad market or a sector ETF. The
    function merges benchmark returns and risk measures by date, then
    computes relative return, volatility, drawdown, slope, beta,
    correlation, and idiosyncratic volatility features.
    """
    bench_cols = [
        "date",
        "ret_1d",
        "ret_20d",
        "ret_60d",
        "ret_120d",
        "volatility_20d",
        "volatility_60d",
        "trailing_max_drawdown_60d",
        "trailing_max_drawdown_252d",
        "log_price_slope_20d",
        "log_price_slope_60d",
    ]
    bench = benchmark_df[bench_cols].copy()
    bench = bench.rename(
        columns={
            "ret_1d": f"{prefix}_ret_1d",
            "ret_20d": f"{prefix}_ret_20d",
            "ret_60d": f"{prefix}_ret_60d",
            "ret_120d": f"{prefix}_ret_120d",
            "volatility_20d": f"{prefix}_volatility_20d",
            "volatility_60d": f"{prefix}_volatility_60d",
            "trailing_max_drawdown_60d": f"{prefix}_trailing_max_drawdown_60d",
            "trailing_max_drawdown_252d": f"{prefix}_trailing_max_drawdown_252d",
            "log_price_slope_20d": f"{prefix}_log_price_slope_20d",
            "log_price_slope_60d": f"{prefix}_log_price_slope_60d",
        }
    )

    df = df.merge(bench, on="date", how="left")

    bench_feature_cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
    df[bench_feature_cols] = df[bench_feature_cols].ffill()

    df[f"rel_ret_20d_vs_{prefix}"] = df["ret_20d"] - df[f"{prefix}_ret_20d"]
    df[f"rel_ret_60d_vs_{prefix}"] = df["ret_60d"] - df[f"{prefix}_ret_60d"]
    df[f"rel_ret_120d_vs_{prefix}"] = df["ret_120d"] - df[f"{prefix}_ret_120d"]

    df[f"vol_ratio_20d_vs_{prefix}"] = df["volatility_20d"] / df[f"{prefix}_volatility_20d"]
    df[f"vol_ratio_60d_vs_{prefix}"] = df["volatility_60d"] / df[f"{prefix}_volatility_60d"]

    df[f"rel_drawdown_60d_vs_{prefix}"] = df["trailing_max_drawdown_60d"] - df[f"{prefix}_trailing_max_drawdown_60d"]
    df[f"rel_drawdown_252d_vs_{prefix}"] = df["trailing_max_drawdown_252d"] - df[f"{prefix}_trailing_max_drawdown_252d"]

    df[f"rel_slope_20d_vs_{prefix}"] = df["log_price_slope_20d"] - df[f"{prefix}_log_price_slope_20d"]
    df[f"rel_slope_60d_vs_{prefix}"] = df["log_price_slope_60d"] - df[f"{prefix}_log_price_slope_60d"]

    df[f"beta_60d_vs_{prefix}"] = (
        df["ret_1d"].rolling(60).cov(df[f"{prefix}_ret_1d"]) /
        df[f"{prefix}_ret_1d"].rolling(60).var()
    )
    df[f"corr_60d_vs_{prefix}"] = df["ret_1d"].rolling(60).corr(df[f"{prefix}_ret_1d"])

    resid = df["ret_1d"] - df[f"beta_60d_vs_{prefix}"] * df[f"{prefix}_ret_1d"]
    df[f"idio_vol_60d_vs_{prefix}"] = resid.rolling(60).std()

    df[f"{prefix}_benchmark_symbol"] = benchmark_symbol
    return df

def main():
    """
    Run the full stage 1 dataset construction pipeline.

    The pipeline loads all raw price files, applies base feature
    engineering, attaches market and sector benchmark features using the
    metadata mappings, creates the future downside label, filters to rows
    with complete modeling inputs, and writes the full dataset, clean
    modeling dataset, and per-symbol summary outputs to disk.
    """
    files = sorted([p for p in RAW_DIR.glob("*.csv") if not p.name.startswith("_")])
    if not files:
        raise ValueError(f"No CSV files found in {RAW_DIR}")

    metadata = load_metadata()
    metadata_map = metadata.set_index("symbol").to_dict(orient="index")

    all_data = {}
    for path in files:
        df = load_price_file(path)
        symbol = df["symbol"].iloc[0]
        df = compute_base_features(df)
        all_data[symbol] = df

    if US_BENCHMARK not in all_data:
        raise ValueError(f"Missing US benchmark file for {US_BENCHMARK}")
    if CA_BENCHMARK not in all_data:
        raise ValueError(f"Missing Canadian benchmark file for {CA_BENCHMARK}")

    rows = []
    for symbol, df in all_data.items():
        if symbol in MARKET_BENCHMARKS:
            continue

        if symbol not in metadata_map:
            continue

        meta = metadata_map[symbol]

        market_symbol = meta["market_benchmark"]
        sector_symbol = meta["sector_benchmark"]

        if market_symbol not in all_data:
            continue
        if sector_symbol not in all_data:
            sector_symbol = market_symbol

        df = df.copy()
        df["country"] = meta["country"]
        df["sector"] = meta["sector"]

        df = add_benchmark_features(df, all_data[market_symbol], prefix="mkt", benchmark_symbol=market_symbol)
        df = add_benchmark_features(df, all_data[sector_symbol], prefix="sector", benchmark_symbol=sector_symbol)

        df["market_benchmark_symbol"] = market_symbol
        df["sector_benchmark_symbol"] = sector_symbol

        df = compute_future_label(df)
        rows.append(df)

    modeling_df = pd.concat(rows, ignore_index=True)

    feature_cols = [
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "ret_60d",
        "ret_120d",
        "volatility_20d",
        "volatility_60d",
        "downside_vol_60d",
        "upside_vol_60d",
        "downside_upside_vol_ratio_60d",
        "dist_from_60d_high",
        "dist_from_252d_high",
        "days_since_60d_high",
        "days_since_252d_high",
        "trailing_max_drawdown_60d",
        "trailing_max_drawdown_252d",
        "avg_volume_20d",
        "avg_volume_60d",
        "volume_spike_20d",
        "avg_dollar_volume_20d",
        "price_to_ma20",
        "price_to_ma60",
        "price_to_ma120",
        "ma20_to_ma60",
        "ma60_to_ma120",
        "log_price_slope_20d",
        "log_price_slope_60d",
        "log_price_slope_120d",
        "trend_accel_20_60",
        "return_skew_20d",
        "return_skew_60d",
        "gap_open_prev_close",
        "intraday_range",
        "avg_intraday_range_20d",
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
        "rel_drawdown_60d_vs_mkt",
        "rel_drawdown_252d_vs_mkt",
        "rel_slope_20d_vs_mkt",
        "rel_slope_60d_vs_mkt",
        "beta_60d_vs_mkt",
        "corr_60d_vs_mkt",
        "idio_vol_60d_vs_mkt",
        "sector_ret_1d",
        "sector_ret_20d",
        "sector_ret_60d",
        "sector_ret_120d",
        "sector_volatility_20d",
        "sector_volatility_60d",
        "rel_ret_20d_vs_sector",
        "rel_ret_60d_vs_sector",
        "rel_ret_120d_vs_sector",
        "vol_ratio_20d_vs_sector",
        "vol_ratio_60d_vs_sector",
        "rel_drawdown_60d_vs_sector",
        "rel_drawdown_252d_vs_sector",
        "rel_slope_20d_vs_sector",
        "rel_slope_60d_vs_sector",
        "beta_60d_vs_sector",
        "corr_60d_vs_sector",
        "idio_vol_60d_vs_sector",
    ]

    keep_cols = [
        "date",
        "symbol",
        "country",
        "sector",
        "market_benchmark_symbol",
        "sector_benchmark_symbol",
        "adjusted_close",
        "volume",
        *feature_cols,
        "future_min_price_60d",
        "future_drawdown_60d",
        "label_drawdown_20pct_60d",
    ]

    modeling_df = modeling_df[keep_cols].sort_values(["date", "symbol"]).reset_index(drop=True)

    clean_df = modeling_df.dropna(subset=feature_cols + ["label_drawdown_20pct_60d"]).copy()
    clean_df["label_drawdown_20pct_60d"] = clean_df["label_drawdown_20pct_60d"].astype(int)

    full_out = OUT_DIR / "stage1_modeling_data_full.csv"
    clean_out = OUT_DIR / "stage1_modeling_data.csv"
    summary_out = OUT_DIR / "stage1_summary.csv"

    modeling_df.to_csv(full_out, index=False)
    clean_df.to_csv(clean_out, index=False)

    summary = (
        clean_df.groupby("symbol")
        .agg(
            n_rows=("symbol", "size"),
            positive_rate=("label_drawdown_20pct_60d", "mean"),
            start_date=("date", "min"),
            end_date=("date", "max"),
        )
        .reset_index()
    )
    summary.to_csv(summary_out, index=False)

    print(f"Saved full dataset to: {full_out}")
    print(f"Saved clean modeling dataset to: {clean_out}")
    print(f"Saved summary to: {summary_out}")
    print()
    print(f"Rows in clean modeling dataset: {len(clean_df)}")
    print(f"Positive rate: {clean_df['label_drawdown_20pct_60d'].mean():.4f}")
    print("Rows by symbol:")
    print(clean_df['symbol'].value_counts().sort_index())

if __name__ == "__main__":
    main()
