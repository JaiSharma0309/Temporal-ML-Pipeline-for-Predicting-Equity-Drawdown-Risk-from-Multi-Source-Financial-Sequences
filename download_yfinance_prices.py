import pandas as pd
import yfinance as yf
from pathlib import Path

SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "SPY",
    "XLK",
    "SHOP.TO",
    "RY.TO",
    "TD.TO",
    "XIU.TO",
    "XIT.TO",
    "XFN.TO",
]

START_DATE = "2016-01-01"
END_DATE = "2025-12-31"

OUTPUT_DIR = Path("data/raw/prices_yfinance")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

    keep_cols = ["date", "symbol", "open", "high", "low", "close", "adjusted_close", "volume"]
    return df[keep_cols].sort_values("date").reset_index(drop=True)


def main():
    successes = []
    failures = []

    for i, symbol in enumerate(SYMBOLS, start=1):
        print(f"[{i}/{len(SYMBOLS)}] Downloading {symbol}...")
        try:
            df = download_one(symbol, START_DATE, END_DATE)
            outpath = OUTPUT_DIR / f"{symbol.replace('/', '_')}.csv"
            df.to_csv(outpath, index=False)
            successes.append(symbol)
            print(f"  saved {symbol} to {outpath} ({len(df)} rows)")
        except Exception as e:
            failures.append((symbol, str(e)))
            print(f"  failed {symbol}: {e}")

    pd.DataFrame({"symbol": successes}).to_csv(OUTPUT_DIR / "_downloaded_ok.csv", index=False)

    if failures:
        pd.DataFrame(failures, columns=["symbol", "error"]).to_csv(
            OUTPUT_DIR / "_download_failures.csv", index=False
        )

    print("\nDone.")
    print(f"Successful: {len(successes)}")
    print(f"Failed: {len(failures)}")


if __name__ == "__main__":
    main()