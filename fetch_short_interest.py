"""
fetch_short_interest.py
=======================
Downloads FINRA bi-monthly short interest position data for US-listed equities
and saves a cleaned, combined parquet file ready for feature engineering.

FINRA publishes consolidated short interest twice a month (settlement dates
around the 15th and last trading day of each month) via their REGSHO program.
Data is fetched via the FINRA REST API (api.finra.org).

Canadian TSX stocks (your TSX 60 universe) are NOT covered by FINRA.
IIROC (now CIRO) publishes equivalent data for Canadian equities at:
  https://www.ciro.ca/investor-support/market-integrity/short-selling-statistics
That data requires manual download; a note is printed at the end of this script.

Usage:
    python fetch_short_interest.py              # fetches 2016–present
    python fetch_short_interest.py --start 2020 # fetches 2020–present

Output:
    data/raw/short_interest/finra_short_interest_raw.parquet
"""

import argparse
import time
import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path("data/raw/short_interest")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FINRA_API_URL = "https://api.finra.org/data/group/OTCMarket/name/consolidatedShortInterest"

PAGE_SIZE     = 5000   # max for synchronous requests
REQUEST_DELAY = 0.5    # seconds between paginated requests
YEAR_DELAY    = 1.0    # seconds between year-level queries


# ── helpers ────────────────────────────────────────────────────────────────────

def fetch_year(year: int, session: requests.Session) -> pd.DataFrame:
    """
    Fetch all consolidated short interest records for a calendar year.
    Paginates automatically using the Record-Total response header.
    """
    start_date = f"{year}-01-01"
    end_date   = f"{year}-12-31"

    frames = []
    offset = 0

    while True:
        payload = {
            "dateRangeFilters": [{
                "fieldName": "settlementDate",
                "startDate": start_date,
                "endDate":   end_date,
            }],
            "limit":  PAGE_SIZE,
            "offset": offset,
        }

        try:
            r = session.post(
                FINRA_API_URL,
                json=payload,
                timeout=60,
                headers={"Accept": "text/plain"},
            )
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"    request error (offset={offset}): {e}")
            break

        text = r.text.strip()
        if not text:
            break

        try:
            df = pd.read_csv(StringIO(text))
        except Exception as e:
            print(f"    parse error (offset={offset}): {e}")
            break

        if df.empty:
            break

        frames.append(df)

        total = int(r.headers.get("Record-Total", 0))
        offset += len(df)

        if offset >= total or len(df) < PAGE_SIZE:
            break

        time.sleep(REQUEST_DELAY)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the FINRA API column names to a consistent schema:
        symbol            — ticker symbol
        settlement_date   — settlement date of the report
        short_interest    — total short interest (shares)
        market            — exchange code
    """
    col_map = {
        "symbolCode":                  "symbol",
        "settlementDate":              "settlement_date",
        "currentShortPositionQuantity": "short_interest",
        "marketClassCode":             "market",
        # fallback / older names
        "Symbol":                      "symbol",
        "ShortInterest":               "short_interest",
        "Market":                      "market",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    required = ["symbol", "short_interest", "settlement_date"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df["symbol"]          = df["symbol"].astype(str).str.upper().str.strip()
    df["short_interest"]  = pd.to_numeric(df["short_interest"], errors="coerce")
    df["settlement_date"] = pd.to_datetime(df["settlement_date"], errors="coerce")
    df = df.dropna(subset=["symbol", "short_interest", "settlement_date"])
    df = df[df["short_interest"] > 0]

    keep = ["symbol", "settlement_date", "short_interest"]
    if "market" in df.columns:
        keep.append("market")
    return df[keep]


# ── main fetch loop ────────────────────────────────────────────────────────────

def fetch_all(start_year: int = 2016, end_year: int | None = None) -> pd.DataFrame:
    if end_year is None:
        end_year = datetime.date.today().year

    session = requests.Session()
    session.headers.update({
        "User-Agent":   "research-data-pipeline/1.0",
        "Content-Type": "application/json",
    })

    frames = []

    for year in range(start_year, end_year + 1):
        print(f"  Fetching {year}...", end=" ", flush=True)
        df = fetch_year(year, session)

        if df.empty:
            print("no data returned")
        else:
            df = standardise_columns(df)
            if df.empty:
                print("no usable columns")
            else:
                n_dates = df["settlement_date"].nunique()
                frames.append(df)
                print(f"{len(df):,} rows, {n_dates} settlement dates")

        time.sleep(YEAR_DELAY)

    if not frames:
        raise RuntimeError(
            "No FINRA data was downloaded. Check your internet connection and "
            "whether the FINRA API URL has changed:\n"
            f"  {FINRA_API_URL}\n"
            "You can also browse the FINRA data catalogue at:\n"
            "  https://developer.finra.org/catalog"
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["symbol", "settlement_date"]).reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["symbol", "settlement_date"], keep="last")

    return combined


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch FINRA short interest data.")
    parser.add_argument("--start", type=int, default=2016,
                        help="First year to fetch (default: 2016)")
    parser.add_argument("--end",   type=int, default=None,
                        help="Last year to fetch (default: current year)")
    args = parser.parse_args()

    print(f"Fetching FINRA consolidated short interest {args.start}–{args.end or 'present'}...")
    print(f"Output: {OUT_DIR / 'finra_short_interest_raw.parquet'}")
    print()

    df = fetch_all(start_year=args.start, end_year=args.end)

    out_path = OUT_DIR / "finra_short_interest_raw.parquet"
    df.to_parquet(out_path, index=False)

    print()
    print(f"Saved {len(df):,} rows → {out_path}")
    print(f"Date range:      {df['settlement_date'].min().date()} → "
          f"{df['settlement_date'].max().date()}")
    print(f"Unique symbols:  {df['symbol'].nunique():,}")
    print(f"Reports/symbol:  {len(df) / df['symbol'].nunique():.1f} on average")

    print()
    print("─" * 60)
    print("NOTE: TSX 60 (Canadian) stocks are NOT covered by FINRA.")
    print("CIRO (formerly IIROC) publishes Canadian short interest at:")
    print("  https://www.ciro.ca/investor-support/market-integrity/")
    print("  short-selling-statistics")
    print("Download those files manually and merge on symbol + date.")
    print("─" * 60)


if __name__ == "__main__":
    main()
