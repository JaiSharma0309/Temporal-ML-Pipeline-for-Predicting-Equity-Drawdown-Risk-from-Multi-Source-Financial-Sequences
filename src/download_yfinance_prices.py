"""
download_yfinance_prices.py
===========================
Builds the universe of equities and downloads historical OHLCV price data
for the S&P 500 + TSX 60 universe plus all benchmark / sector-ETF tickers.

Steps
-----
  1. Scrape current S&P 500 and TSX 60 constituents from Wikipedia.
  2. Map each stock to its country, GICS sector, market benchmark (SPY / XIU.TO)
     and sector ETF benchmark (e.g. XLK for US Tech, XIT.TO for CA Tech).
  3. Save the full universe metadata to data/metadata/.
  4. Download daily price history (2016–2025) via yfinance for every stock
     plus the benchmark and sector ETFs, saving one CSV per ticker.

Run:
    python download_yfinance_prices.py

Output:
    data/metadata/equity_universe_metadata.csv   — one row per ticker
    data/raw/prices_yfinance/{SYMBOL}.csv        — OHLCV + adjusted close per ticker
    data/raw/prices_yfinance/_downloaded_ok.csv  — list of successful downloads
    data/raw/prices_yfinance/_download_failures.csv — failures (if any)

Note: constituent lists are scraped live from Wikipedia, so they reflect the
current index membership.  This introduces survivorship bias — stocks that were
delisted after a major drawdown are absent from the training data.
"""

from pathlib import Path
from io import StringIO
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

START_DATE = "2016-01-01"
END_DATE   = "2025-12-31"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

RAW_DIR  = REPO_ROOT / "data/raw/prices_yfinance"
META_DIR = REPO_ROOT / "data/metadata"
RAW_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

# Wikipedia pages used to scrape constituent lists
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
TSX60_URL = "https://en.wikipedia.org/wiki/S%26P/TSX_60"

# Broad market benchmarks — used as market-relative feature denominators
US_BENCHMARK = "SPY"
CA_BENCHMARK = "XIU.TO"

# Request header to identify the scraper to Wikipedia's servers
HEADERS = {"User-Agent": "Temporal-Equity-Drawdown-Risk-Project/1.0"}

# GICS sector → US sector ETF (SPDR Select Sector series)
US_SECTOR_ETF_MAP = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":        "XLP",
    "Energy":                  "XLE",
    "Financials":              "XLF",
    "Health Care":             "XLV",
    "Industrials":             "XLI",
    "Information Technology":  "XLK",
    "Materials":               "XLB",
    "Real Estate":             "XLRE",
    "Utilities":               "XLU",
}

# GICS sector → Canadian sector ETF (iShares series)
# Some sector names differ slightly between Wikipedia's US and CA tables,
# so both variants are mapped to the same ETF.
CA_SECTOR_ETF_MAP = {
    "Information Technology": "XIT.TO",
    "Financial Services":     "XFN.TO",
    "Financials":             "XFN.TO",
    "Energy":                 "XEG.TO",
    "Basic Materials":        "XMA.TO",
    "Materials":              "XMA.TO",
    "Utilities":              "XUT.TO",
    "Real Estate":            "XRE.TO",
    "Consumer Staples":       "XST.TO",
    "Communication Services": "XTL.TO",
    "Industrials":            "XIN.TO",
    "Health Care":            "XHC.TO",
}

# All benchmark + sector ETF tickers that must be downloaded alongside equities
EXTRA_SYMBOLS = sorted(
    {US_BENCHMARK, CA_BENCHMARK}
    | set(US_SECTOR_ETF_MAP.values())
    | set(CA_SECTOR_ETF_MAP.values())
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def read_html_with_headers(url: str):
    """Fetch a web page with a custom User-Agent and parse all HTML tables."""
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))


def to_yfinance_us_ticker(symbol: str) -> str:
    """Convert an S&P 500 ticker to yfinance format (dots → hyphens, e.g. BRK.B → BRK-B)."""
    return symbol.replace(".", "-").strip()


def to_yfinance_ca_ticker(symbol: str) -> str:
    """Append .TO suffix to TSX tickers if not already present (required by yfinance)."""
    symbol = symbol.strip()
    if symbol.endswith(".TO"):
        return symbol
    return f"{symbol}.TO"


def clean_sector_name(sector: str) -> str:
    """Normalise a sector string — returns 'Unknown' for NaN values."""
    if pd.isna(sector):
        return "Unknown"
    return str(sector).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Universe construction
# ══════════════════════════════════════════════════════════════════════════════

def fetch_sp500_constituents() -> pd.DataFrame:
    """
    Scrape the current S&P 500 constituent list from Wikipedia.

    Maps each ticker to its GICS sector, market benchmark (SPY), and the
    corresponding SPDR sector ETF benchmark.  Stocks with no sector ETF
    mapping fall back to SPY.

    Returns a DataFrame with columns:
        symbol, raw_symbol, company, sector, sub_industry,
        country, market_benchmark, sector_benchmark
    """
    tables = read_html_with_headers(SP500_URL)
    df = tables[0].copy()  # first table on the page is always the constituent list

    df = df.rename(columns={
        "Symbol":           "raw_symbol",
        "Security":         "company",
        "GICS Sector":      "sector",
        "GICS Sub-Industry": "sub_industry",
    })

    df["country"]          = "US"
    df["sector"]           = df["sector"].apply(clean_sector_name)
    df["symbol"]           = df["raw_symbol"].astype(str).apply(to_yfinance_us_ticker)
    df["market_benchmark"] = US_BENCHMARK
    # Fall back to SPY for any sector not in the map (e.g. "Unknown")
    df["sector_benchmark"] = df["sector"].map(US_SECTOR_ETF_MAP).fillna(US_BENCHMARK)

    keep_cols = ["symbol", "raw_symbol", "company", "sector", "sub_industry",
                 "country", "market_benchmark", "sector_benchmark"]
    return df[keep_cols].drop_duplicates(subset=["symbol"]).reset_index(drop=True)


def fetch_tsx60_constituents() -> pd.DataFrame:
    """
    Scrape the current TSX 60 constituent list from Wikipedia.

    Searches all tables on the page for one containing Symbol / Company / Sector
    columns, which is more robust than relying on table index (page layout changes).

    Returns the same column schema as fetch_sp500_constituents.
    """
    tables = read_html_with_headers(TSX60_URL)

    # Find the first table that has the expected columns
    candidate = None
    for t in tables:
        cols = {str(c).strip() for c in t.columns}
        if {"Symbol", "Company", "Sector"}.issubset(cols):
            candidate = t.copy()
            break

    if candidate is None:
        raise ValueError("Could not find TSX 60 constituents table on the page.")

    df = candidate.rename(columns={
        "Symbol":  "raw_symbol",
        "Company": "company",
        "Sector":  "sector",
    })

    df["country"]          = "CA"
    df["sector"]           = df["sector"].apply(clean_sector_name)
    df["symbol"]           = df["raw_symbol"].astype(str).apply(to_yfinance_ca_ticker)
    df["market_benchmark"] = CA_BENCHMARK
    df["sector_benchmark"] = df["sector"].map(CA_SECTOR_ETF_MAP).fillna(CA_BENCHMARK)
    df["sub_industry"]     = np.nan  # Wikipedia's TSX table has no sub-industry column

    keep_cols = ["symbol", "raw_symbol", "company", "sector", "sub_industry",
                 "country", "market_benchmark", "sector_benchmark"]
    return df[keep_cols].drop_duplicates(subset=["symbol"]).reset_index(drop=True)


def build_universe_metadata() -> pd.DataFrame:
    """
    Combine S&P 500 and TSX 60 into a single universe DataFrame.

    Deduplicates on symbol in case a stock appears in both indices (rare but
    possible for cross-listed names), then sorts for deterministic output.
    """
    sp500 = fetch_sp500_constituents()
    tsx60 = fetch_tsx60_constituents()

    universe = pd.concat([sp500, tsx60], ignore_index=True)
    universe = universe.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    universe = universe.sort_values(["country", "symbol"]).reset_index(drop=True)
    return universe


# ══════════════════════════════════════════════════════════════════════════════
# Price download
# ══════════════════════════════════════════════════════════════════════════════

def download_one(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download daily OHLCV + adjusted close for a single ticker via yfinance.

    auto_adjust=False is intentional: we keep both Close and Adj Close so
    downstream feature engineering can use the unadjusted close for
    intraday range calculations while using adjusted_close for returns.

    Returns a DataFrame sorted by date with columns:
        date, symbol, open, high, low, close, adjusted_close, volume
    """
    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"{symbol}: no data returned")

    # yfinance sometimes returns a MultiIndex when downloading a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df["symbol"] = symbol

    df = df.rename(columns={
        "Date":      "date",
        "Open":      "open",
        "High":      "high",
        "Low":       "low",
        "Close":     "close",
        "Adj Close": "adjusted_close",
        "Volume":    "volume",
    })

    keep_cols = ["date", "symbol", "open", "high", "low", "close",
                 "adjusted_close", "volume"]
    return df[keep_cols].sort_values("date").reset_index(drop=True)


def download_symbols(symbols, start_date: str, end_date: str, output_dir: Path):
    """
    Download price history for a list of symbols, saving one CSV per ticker.

    Failures are caught individually so a single bad ticker doesn't abort the
    entire run.  Summary CSVs (_downloaded_ok.csv, _download_failures.csv) are
    written at the end for easy re-run of failed tickers.

    Returns (successes, failures) where failures is a list of (symbol, error) tuples.
    """
    successes = []
    failures  = []

    for i, symbol in enumerate(symbols, start=1):
        print(f"[{i}/{len(symbols)}] Downloading {symbol}...", end="  ", flush=True)
        try:
            df = download_one(symbol, start_date, end_date)
            # Replace '/' in symbol names to avoid creating subdirectories
            outpath = output_dir / f"{symbol.replace('/', '_')}.csv"
            df.to_csv(outpath, index=False)
            successes.append(symbol)
            print(f"saved {len(df)} rows → {outpath.name}")
        except Exception as e:
            failures.append((symbol, str(e)))
            print(f"FAILED: {e}")

        time.sleep(0.2)  # be polite to Yahoo Finance's rate limiter

    # Write manifest files for easy auditing / retry of failures
    pd.DataFrame({"symbol": successes}).to_csv(
        output_dir / "_downloaded_ok.csv", index=False
    )
    if failures:
        pd.DataFrame(failures, columns=["symbol", "error"]).to_csv(
            output_dir / "_download_failures.csv", index=False
        )

    return successes, failures


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Orchestrate universe construction and price download.

    Downloads the full universe (equities + benchmarks + sector ETFs).
    Benchmarks and sector ETFs are included because they are used as feature
    denominators in the ML pipeline (market-relative returns, beta, etc.).
    """
    # Build and save universe metadata first
    universe = build_universe_metadata()
    universe.to_csv(META_DIR / "equity_universe_metadata.csv", index=False)

    # Include benchmark + sector ETF tickers alongside the equity universe
    price_symbols = sorted(set(universe["symbol"].tolist()) | set(EXTRA_SYMBOLS))

    print(f"Equities in universe: {len(universe)}")
    print(f"Total symbols to download (incl. benchmarks/sector ETFs): {len(price_symbols)}")
    print()
    print("Universe counts by country:")
    print(universe["country"].value_counts())
    print()

    successes, failures = download_symbols(price_symbols, START_DATE, END_DATE, RAW_DIR)

    print(f"\nDone.  Successful: {len(successes)}  Failed: {len(failures)}")
    if failures:
        print(f"Check failed tickers: {RAW_DIR / '_download_failures.csv'}")


if __name__ == "__main__":
    main()
