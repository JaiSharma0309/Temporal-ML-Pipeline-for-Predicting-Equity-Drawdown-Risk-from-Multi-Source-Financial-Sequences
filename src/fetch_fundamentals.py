"""
fetch_fundamentals.py
=====================
Downloads quarterly fundamental data (income statement + balance sheet) for
your S&P 500 + TSX 60 universe via yfinance and builds a time-aware feature
dataset ready for merging into the stage-1 modeling pipeline.

Key design decisions
--------------------
1. REPORTING LAG: Earnings are reported ~45 days after the quarter ends.
   We forward-shift all quarterly data by 45 days before joining to daily
   price rows, so the model only sees information that was actually *public*
   on each prediction date.  Without this, rows in e.g. April would see
   Q1 earnings that weren't released until May — lookahead leakage.

2. FORWARD FILL: Fundamentals change quarterly; we forward-fill within each
   stock so every daily row gets the most recently *available* report.

3. POINT-IN-TIME P/E: yfinance's t.info["trailingPE"] is current-only, not
   historical.  We skip it.  Instead we compute a price-relative-to-earnings
   proxy from available statement data where possible.  For a proper
   historical P/E time series you would need Compustat/WRDS.

4. GRACEFUL FAILURES: Many tickers fail (delistings, data gaps, rate limits).
   We log failures and continue; missing tickers get NaN in the feature columns
   and are handled by SimpleImputer in the ML pipeline.

Usage:
    # Read symbol list from your processed data file
    python fetch_fundamentals.py

    # Override symbol list from a file (one ticker per line)
    python fetch_fundamentals.py --symbols-file my_tickers.txt

    # Re-fetch even if output already exists
    python fetch_fundamentals.py --overwrite

Output:
    data/raw/fundamentals/fundamentals_features.parquet
        Columns: symbol, report_available_date, revenue_growth_yoy,
                 revenue_growth_decel, gross_margin, operating_margin,
                 debt_to_equity, interest_coverage, current_ratio
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance not installed.  Run: pip install yfinance")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

DATA_PATH  = REPO_ROOT / "data/processed/stage1_modeling_data.csv"
OUT_DIR    = REPO_ROOT / "data/raw/fundamentals"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH   = OUT_DIR / "fundamentals_features.parquet"

# Earnings are typically reported 30–60 days after quarter end.
# 45-day lag is a conservative middle ground that avoids lookahead on ~95%
# of S&P 500 earnings while not being so conservative that we miss the signal.
REPORTING_LAG_DAYS = 45

REQUEST_DELAY = 0.5   # seconds between tickers — yfinance has rate limits


# ── per-ticker data fetch ──────────────────────────────────────────────────────

def fetch_ticker_statements(symbol: str) -> dict:
    """
    Fetch quarterly income statement and balance sheet for one ticker.
    Returns a dict with keys 'income' and 'balance' (DataFrames, index=metric,
    columns=quarter-end dates) or empty DataFrames on failure.
    """
    empty = {"income": pd.DataFrame(), "balance": pd.DataFrame()}
    try:
        t = yf.Ticker(symbol)
        income  = t.quarterly_income_stmt
        balance = t.quarterly_balance_sheet
        if income is None or income.empty:
            return empty
        return {
            "income":  income,
            "balance": balance if balance is not None else pd.DataFrame(),
        }
    except Exception as e:
        return empty


# ── feature computation for one ticker ────────────────────────────────────────

def compute_ticker_features(symbol: str, data: dict) -> pd.DataFrame:
    """
    Turn raw yfinance statement DataFrames into a time-indexed feature DataFrame
    for one ticker.  Returns rows keyed by `report_available_date` (= quarter
    end + REPORTING_LAG_DAYS), which is when the market could first see this data.

    Features computed:
      revenue_growth_yoy    — quarterly revenue vs same quarter 1 year ago
      revenue_growth_decel  — change in QoQ growth rate (negative = slowing)
      gross_margin          — gross profit / revenue
      operating_margin      — operating income / revenue
      debt_to_equity        — total debt / stockholders' equity
      interest_coverage     — EBIT / interest expense  (>2 = healthy, <1 = danger)
      current_ratio         — current assets / current liabilities
    """
    income  = data.get("income",  pd.DataFrame())
    balance = data.get("balance", pd.DataFrame())

    if income.empty:
        return pd.DataFrame()

    # yfinance: index = metric name, columns = quarter-end date
    # Transpose so rows = quarters, sort chronologically
    try:
        inc = income.T.sort_index()
        inc.index = pd.to_datetime(inc.index)
    except Exception:
        return pd.DataFrame()

    bal = pd.DataFrame()
    if not balance.empty:
        try:
            bal = balance.T.sort_index()
            bal.index = pd.to_datetime(bal.index)
        except Exception:
            pass

    rows = []

    for i, qdate in enumerate(inc.index):
        row = {
            "symbol":               symbol,
            "quarter_end":          qdate,
            "report_available_date": qdate + pd.Timedelta(days=REPORTING_LAG_DAYS),
        }

        # ── Revenue features ──────────────────────────────────────────────────
        rev_col = next(
            (c for c in ["Total Revenue", "Revenue", "TotalRevenue"]
             if c in inc.columns), None
        )
        if rev_col:
            rev = inc[rev_col]
            rev_now = rev.iloc[i]

            # YoY growth: compare to same quarter 4 periods ago
            if i >= 4:
                rev_1y = rev.iloc[i - 4]
                if pd.notna(rev_now) and pd.notna(rev_1y) and rev_1y != 0:
                    row["revenue_growth_yoy"] = float((rev_now - rev_1y) / abs(rev_1y))

            # QoQ deceleration: current QoQ minus prior QoQ
            # Negative value = growth is slowing — a key drawdown precursor
            if i >= 2:
                r0 = rev.iloc[i]
                r1 = rev.iloc[i - 1]
                r2 = rev.iloc[i - 2]
                if all(pd.notna([r0, r1, r2])) and r1 != 0 and r2 != 0:
                    qoq_now  = (r0 - r1) / abs(r1)
                    qoq_prev = (r1 - r2) / abs(r2)
                    row["revenue_growth_decel"] = float(qoq_now - qoq_prev)

        # ── Margin features ───────────────────────────────────────────────────
        gross_col = next(
            (c for c in ["Gross Profit", "GrossProfit"] if c in inc.columns), None
        )
        op_col = next(
            (c for c in ["Operating Income", "EBIT", "OperatingIncome",
                          "Operating Income Loss"]
             if c in inc.columns), None
        )
        rev_val = inc[rev_col].iloc[i] if rev_col else np.nan

        if gross_col and pd.notna(rev_val) and rev_val != 0:
            gp = inc[gross_col].iloc[i]
            if pd.notna(gp):
                row["gross_margin"] = float(gp / rev_val)

        if op_col and pd.notna(rev_val) and rev_val != 0:
            oi = inc[op_col].iloc[i]
            if pd.notna(oi):
                row["operating_margin"] = float(oi / rev_val)

        # ── Balance sheet features ────────────────────────────────────────────
        if not bal.empty:
            # Find the balance sheet row closest to (but not after) this quarter
            bal_dates_before = bal.index[bal.index <= qdate]
            if len(bal_dates_before) > 0:
                b = bal.loc[bal_dates_before[-1]]

                # Debt / equity
                debt_col   = next(
                    (c for c in ["Total Debt", "TotalDebt", "Long Term Debt",
                                  "LongTermDebt"]
                     if c in bal.columns), None
                )
                equity_col = next(
                    (c for c in ["Stockholders Equity", "StockholdersEquity",
                                  "Common Stock Equity", "Total Equity Gross Minority Interest"]
                     if c in bal.columns), None
                )
                if debt_col and equity_col:
                    debt   = b.get(debt_col, np.nan)
                    equity = b.get(equity_col, np.nan)
                    if pd.notna(debt) and pd.notna(equity) and equity != 0:
                        row["debt_to_equity"] = float(debt / abs(equity))

                # Interest coverage: EBIT / interest expense
                int_col = next(
                    (c for c in ["Interest Expense", "InterestExpense",
                                  "Interest Expense Non Operating"]
                     if c in inc.columns), None
                )
                if op_col and int_col:
                    ebit = inc[op_col].iloc[i]
                    ie   = inc[int_col].iloc[i]
                    # Interest expense in yfinance is often negative (cash outflow)
                    ie_abs = abs(ie) if pd.notna(ie) else np.nan
                    if pd.notna(ebit) and pd.notna(ie_abs) and ie_abs > 0:
                        row["interest_coverage"] = float(ebit / ie_abs)

                # Current ratio: current assets / current liabilities
                ca_col = next(
                    (c for c in ["Current Assets", "CurrentAssets",
                                  "Total Current Assets"]
                     if c in bal.columns), None
                )
                cl_col = next(
                    (c for c in ["Current Liabilities", "CurrentLiabilities",
                                  "Total Current Liabilities"]
                     if c in bal.columns), None
                )
                if ca_col and cl_col:
                    ca = b.get(ca_col, np.nan)
                    cl = b.get(cl_col, np.nan)
                    if pd.notna(ca) and pd.notna(cl) and cl != 0:
                        row["current_ratio"] = float(ca / abs(cl))

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ── build full feature dataset ─────────────────────────────────────────────────

def build_fundamental_features(symbols: list[str]) -> pd.DataFrame:
    """
    Fetch statements for all symbols, compute features, and combine into a
    single DataFrame indexed by (symbol, report_available_date).
    """
    all_frames = []
    n = len(symbols)
    failed = []

    for i, sym in enumerate(symbols, 1):
        print(f"  [{i:>4}/{n}]  {sym:<8}", end="  ", flush=True)
        data = fetch_ticker_statements(sym)

        if data["income"].empty:
            print("✗ no data")
            failed.append(sym)
        else:
            df_sym = compute_ticker_features(sym, data)
            if df_sym.empty:
                print("✗ feature error")
                failed.append(sym)
            else:
                all_frames.append(df_sym)
                n_rows = len(df_sym)
                print(f"✓ {n_rows} quarters")

        time.sleep(REQUEST_DELAY)

    if not all_frames:
        raise RuntimeError("No fundamental data fetched for any ticker.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["symbol", "report_available_date"])

    print()
    print(f"  Fetched: {len(symbols) - len(failed)}/{len(symbols)} tickers")
    if failed:
        print(f"  Failed ({len(failed)}): {', '.join(failed[:20])}"
              + (" ..." if len(failed) > 20 else ""))

    return combined


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch fundamental features via yfinance.")
    parser.add_argument("--symbols-file", type=str, default=None,
                        help="Path to text file with one ticker per line. "
                             "Default: reads symbols from stage1_modeling_data.csv")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-fetch even if output parquet already exists")
    args = parser.parse_args()

    if OUT_PATH.exists() and not args.overwrite:
        print(f"Output already exists: {OUT_PATH}")
        print("Use --overwrite to re-fetch.  Exiting.")
        return

    # ── load symbol list ───────────────────────────────────────────────────────
    if args.symbols_file:
        symbols = Path(args.symbols_file).read_text().strip().splitlines()
        symbols = [s.strip().upper() for s in symbols if s.strip()]
    else:
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Could not find {DATA_PATH}.  Either run the data pipeline first "
                "or pass --symbols-file with a list of tickers."
            )
        sym_df  = pd.read_csv(DATA_PATH, usecols=["symbol", "country"])
        # yfinance uses different suffixes for TSX stocks: BNS → BNS.TO
        def to_yf_symbol(row):
            if str(row.get("country", "")).upper() in ("CA", "CAN", "CANADA"):
                s = str(row["symbol"])
                return s if s.endswith(".TO") else s + ".TO"
            return str(row["symbol"])
        symbols = sym_df.apply(to_yf_symbol, axis=1).drop_duplicates().tolist()

    print(f"Fetching fundamentals for {len(symbols)} symbols...")
    print(f"Output: {OUT_PATH}")
    print()

    features_df = build_fundamental_features(symbols)

    # Keep symbols exactly as fetched (e.g., TSX names retain ".TO") so joins
    # in train_drawdown_risk_models.py align with stage1_modeling_data.csv symbols.

    features_df.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved {len(features_df):,} rows → {OUT_PATH}")
    feat_cols = [c for c in features_df.columns
                 if c not in ("symbol", "quarter_end", "report_available_date")]
    print(f"Feature columns: {feat_cols}")
    coverage = features_df[feat_cols].notna().mean().round(3)
    print("\nColumn coverage (fraction non-null):")
    print(coverage.to_string())


if __name__ == "__main__":
    main()
