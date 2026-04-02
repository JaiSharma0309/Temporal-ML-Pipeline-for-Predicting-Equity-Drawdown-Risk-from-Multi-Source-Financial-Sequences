from pathlib import Path
from io import StringIO
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

START_DATE = "2016-01-01"
END_DATE = "2025-12-31"

RAW_DIR = Path("data/raw/prices_yfinance")
META_DIR = Path("data/metadata")
RAW_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
TSX60_URL = "https://en.wikipedia.org/wiki/S%26P/TSX_60"

US_BENCHMARK = "SPY"
CA_BENCHMARK = "XIU.TO"

HEADERS = {
    "User-Agent": "Temporal-Equity-Drawdown-Risk-Project/1.0"
}

US_SECTOR_ETF_MAP = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

CA_SECTOR_ETF_MAP = {
    "Information Technology": "XIT.TO",
    "Financial Services": "XFN.TO",
    "Financials": "XFN.TO",
    "Energy": "XEG.TO",
    "Basic Materials": "XMA.TO",
    "Materials": "XMA.TO",
    "Utilities": "XUT.TO",
    "Real Estate": "XRE.TO",
    "Consumer Staples": "XST.TO",
    "Communication Services": "XTL.TO",
    "Industrials": "XIN.TO",
    "Health Care": "XHC.TO",
}

EXTRA_SYMBOLS = sorted(
    {US_BENCHMARK, CA_BENCHMARK}
    | set(US_SECTOR_ETF_MAP.values())
    | set(CA_SECTOR_ETF_MAP.values())
)


def read_html_with_headers(url: str):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))


def to_yfinance_us_ticker(symbol: str) -> str:
    return symbol.replace(".", "-").strip()


def to_yfinance_ca_ticker(symbol: str) -> str:
    symbol = symbol.strip()
    if symbol.endswith(".TO"):
        return symbol
    return f"{symbol}.TO"


def clean_sector_name(sector: str) -> str:
    if pd.isna(sector):
        return "Unknown"
    return str(sector).strip()


def fetch_sp500_constituents() -> pd.DataFrame:
    tables = read_html_with_headers(SP500_URL)
    df = tables[0].copy()

    df = df.rename(columns={
        "Symbol": "raw_symbol",
        "Security": "company",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
    })

    df["country"] = "US"
    df["sector"] = df["sector"].apply(clean_sector_name)
    df["symbol"] = df["raw_symbol"].astype(str).apply(to_yfinance_us_ticker)
    df["market_benchmark"] = US_BENCHMARK
    df["sector_benchmark"] = df["sector"].map(US_SECTOR_ETF_MAP).fillna(US_BENCHMARK)

    keep_cols = [
        "symbol",
        "raw_symbol",
        "company",
        "sector",
        "sub_industry",
        "country",
        "market_benchmark",
        "sector_benchmark",
    ]
    return df[keep_cols].drop_duplicates(subset=["symbol"]).reset_index(drop=True)


def fetch_tsx60_constituents() -> pd.DataFrame:
    tables = read_html_with_headers(TSX60_URL)

    candidate = None
    for t in tables:
        cols = {str(c).strip() for c in t.columns}
        if {"Symbol", "Company", "Sector"}.issubset(cols):
            candidate = t.copy()
            break

    if candidate is None:
        raise ValueError("Could not find TSX 60 constituents table on the page.")

    df = candidate.rename(columns={
        "Symbol": "raw_symbol",
        "Company": "company",
        "Sector": "sector",
    })

    df["country"] = "CA"
    df["sector"] = df["sector"].apply(clean_sector_name)
    df["symbol"] = df["raw_symbol"].astype(str).apply(to_yfinance_ca_ticker)
    df["market_benchmark"] = CA_BENCHMARK
    df["sector_benchmark"] = df["sector"].map(CA_SECTOR_ETF_MAP).fillna(CA_BENCHMARK)
    df["sub_industry"] = np.nan

    keep_cols = [
        "symbol",
        "raw_symbol",
        "company",
        "sector",
        "sub_industry",
        "country",
        "market_benchmark",
        "sector_benchmark",
    ]
    return df[keep_cols].drop_duplicates(subset=["symbol"]).reset_index(drop=True)


def build_universe_metadata() -> pd.DataFrame:
    sp500 = fetch_sp500_constituents()
    tsx60 = fetch_tsx60_constituents()

    universe = pd.concat([sp500, tsx60], ignore_index=True)
    universe = universe.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    universe = universe.sort_values(["country", "symbol"]).reset_index(drop=True)
    return universe


def download_one(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"{symbol}: no data returned")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df["symbol"] = symbol

    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjusted_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    keep_cols = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]
    return df[keep_cols].sort_values("date").reset_index(drop=True)


def download_symbols(symbols, start_date, end_date, output_dir: Path):
    successes = []
    failures = []

    for i, symbol in enumerate(symbols, start=1):
        print(f"[{i}/{len(symbols)}] Downloading {symbol}...")
        try:
            df = download_one(symbol, start_date, end_date)
            outpath = output_dir / f"{symbol.replace('/', '_')}.csv"
            df.to_csv(outpath, index=False)
            successes.append(symbol)
            print(f"  saved {symbol} to {outpath} ({len(df)} rows)")
        except Exception as e:
            failures.append((symbol, str(e)))
            print(f"  failed {symbol}: {e}")

        time.sleep(0.2)

    pd.DataFrame({"symbol": successes}).to_csv(output_dir / "_downloaded_ok.csv", index=False)

    if failures:
        pd.DataFrame(failures, columns=["symbol", "error"]).to_csv(
            output_dir / "_download_failures.csv", index=False
        )

    return successes, failures


def main():
    universe = build_universe_metadata()
    universe.to_csv(META_DIR / "equity_universe_metadata.csv", index=False)

    price_symbols = sorted(set(universe["symbol"].tolist()) | set(EXTRA_SYMBOLS))

    print(f"Equities in universe: {len(universe)}")
    print(f"Total symbols to download including benchmarks/sector ETFs: {len(price_symbols)}")
    print()
    print("Universe counts by country:")
    print(universe["country"].value_counts())
    print()

    successes, failures = download_symbols(price_symbols, START_DATE, END_DATE, RAW_DIR)

    print("\nDone.")
    print(f"Successful downloads: {len(successes)}")
    print(f"Failed downloads: {len(failures)}")

    if failures:
        print("\nSome symbols failed. Check:")
        print(RAW_DIR / "_download_failures.csv")


if __name__ == "__main__":
    main()