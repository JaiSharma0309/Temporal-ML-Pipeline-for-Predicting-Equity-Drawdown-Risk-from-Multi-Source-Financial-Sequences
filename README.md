# Temporal ML Pipeline for Equity Drawdown Risk

Time-aware machine learning pipeline for predicting whether an equity will experience a **20%+ drawdown in the next 60 trading days**.

This project builds a multi-source dataset from price action, benchmark-relative features, short interest, and fundamentals; trains multiple time-aware models; and evaluates both statistical quality and practical ranking value.

## Project Metadata

- **Author:** Jai Sharma
- **License:** MIT
- **Status:** Research project with end-to-end reproducible pipeline
- **Primary language:** Python

## 10-Second Snapshot

- Test top-decile lift (random forest): **2.31x**
- Best walk-forward classifier by PR AUC: `ensemble_rf_lr`
- Highest-risk decile is materially enriched for future 20%+ drawdowns

![Test Decile Event Rate](results/stage1/plots/04_test_decile_event_rate.png)

![Capture vs Workload](results/stage1/plots/05_test_capture_vs_workload.png)

## Purpose

This repo answers a practical risk-screening question:

- Which stocks are most likely to suffer a major drawdown soon?
- Can we rank a universe of stocks so risk teams can focus on the highest-risk names first?

The target is:

- `label_drawdown_20pct_60d` = 1 if the stock falls 20% or more at any point in the next 60 trading days, else 0

## What This Repository Includes

- Universe construction from S&P 500 and TSX 60 constituents
- Daily price ingestion and benchmark mapping
- Feature engineering for technical, relative, short-interest, and fundamental signals
- Leakage-aware temporal model training and validation
- Business-facing evaluation plots and summary outputs

## File Structure

```text
.
├── LICENSE
├── CONTRIBUTING.md
├── Makefile
├── README.md
├── requirements.txt
├── data/
│   ├── metadata/
│   │   └── equity_universe_metadata.csv
│   └── processed/
│       ├── stage1_modeling_data_reduced.xlsx
│       └── stage1_summary.csv
├── results/
│   └── stage1/
│       ├── best_model.txt
│       ├── random_forest_selected_threshold.txt
│       ├── plots/
│       ├── reports/
│       └── tables/
└── src/
    ├── build_modeling_dataset.py
    ├── download_yfinance_prices.py
    ├── fetch_fundamentals.py
    ├── fetch_short_interest.py
    ├── model_visualizations.py
    └── train_drawdown_risk_models.py
```

## Repository Structure

- `src/download_yfinance_prices.py`  
  Builds the equity universe and downloads OHLCV price data for stocks, benchmarks, and sector ETFs.

- `src/build_modeling_dataset.py`  
  Creates technical and benchmark-relative features, computes forward drawdown labels, and writes clean modeling datasets.

- `src/fetch_short_interest.py`  
  Downloads FINRA short-interest data for US-listed equities and saves a cleaned parquet file.

- `src/fetch_fundamentals.py`  
  Downloads quarterly fundamentals via yfinance, applies reporting-lag logic, and outputs point-in-time-safe features.

- `src/train_drawdown_risk_models.py`  
  Merges all feature blocks, applies temporal safeguards, runs walk-forward CV and final split evaluation, and writes metrics and artifacts.

- `src/model_visualizations.py`  
  Creates performance visuals and a short business-impact summary.

## Data Pipeline

### 1. Download universe and prices

Run:

```bash
python src/download_yfinance_prices.py
```

This step:

- scrapes live S&P 500 and TSX 60 constituents,
- maps each symbol to country, sector, market benchmark, and sector benchmark ETF,
- downloads daily prices from `2016-01-01` to `2025-12-31`.

Outputs:

- `data/metadata/equity_universe_metadata.csv`
- `data/raw/prices_yfinance/*.csv`
- `_downloaded_ok.csv`
- `_download_failures.csv`

### 2. Build the modeling dataset

Run:

```bash
python src/build_modeling_dataset.py
```

This step:

- loads raw per-symbol OHLCV files,
- engineers base and benchmark-relative features,
- computes the forward 60-day drawdown target,
- exports full and clean training datasets.

Outputs:

- `data/processed/stage1_modeling_data_full.csv`
- `data/processed/stage1_modeling_data.csv`
- `data/processed/stage1_summary.csv`

Note: `data/processed/stage1_modeling_data.csv` is intentionally not committed because it is large and reproducible.

### 3. Optional short-interest features

Run:

```bash
python src/fetch_short_interest.py
```

Output:

- `data/raw/short_interest/finra_short_interest_raw.parquet`

### 4. Optional fundamentals features

Run:

```bash
python src/fetch_fundamentals.py --overwrite
```

Output:

- `data/raw/fundamentals/fundamentals_features.parquet`

## Modeling Approach

`src/train_drawdown_risk_models.py` includes:

- Feature blocks:
  - 70 base numeric features
  - regime flags
  - cross-sectional rank features
  - optional short-interest features
  - optional fundamental features
- Temporal safeguards:
  - training boundary embargo
  - overlapping-label purge
  - expanding walk-forward cross-validation
- Model families:
  - Classifiers: Dummy, Logistic Regression, Random Forest, HistGradientBoosting, RF+LR ensemble
  - Regressors: Ridge, RF Regressor, HGB Regressor

## Current Results

Data and features:

- Rows: `1,033,811`
- Numeric features: `105`

Coverage diagnostics:

- Short interest coverage:
  - Overall: `87.80%`
  - US: `97.30%`
  - CA: `0.00%`
- Fundamentals coverage:
  - Overall: `8.44%`
  - US: `8.43%`
  - CA: `8.53%`

Walk-forward CV:

- Best average classifier by PR AUC: `ensemble_rf_lr`
- ROC AUC: `0.6799`
- PR AUC: `0.1636`
- lift@10%: `2.6206x`

Final split:

- Best classifier by validation PR AUC: `random_forest`
- Test ROC AUC: `0.6826`
- Test PR AUC: `0.1678`
- Test lift@10%: `2.31x`

Interpretation: the model is meaningfully better than random ranking for drawdown screening and shows strong top-decile enrichment.

## Visualizations

Run:

```bash
python src/model_visualizations.py
```

Generates:

- `results/stage1/plots/01_top_decile_lift_by_fold.png`
- `results/stage1/plots/02_fold_by_fold_roc_pr.png`
- `results/stage1/plots/03_test_cumulative_lift_curve.png`
- `results/stage1/plots/04_test_decile_event_rate.png`
- `results/stage1/plots/05_test_capture_vs_workload.png`
- `results/stage1/reports/business_impact_summary.md`

## Quickstart

From a fresh clone:

```bash
python -m pip install -r requirements.txt
make pipeline
```

If data already exists locally:

```bash
make train
make visuals
```

## Full Run Order

```bash
# 1. Build universe + prices
python src/download_yfinance_prices.py

# 2. Build modeling dataset
python src/build_modeling_dataset.py

# 3. Optional: short interest
python src/fetch_short_interest.py

# 4. Optional: fundamentals
python src/fetch_fundamentals.py --overwrite

# 5. Train and evaluate
python src/train_drawdown_risk_models.py

# 6. Detailed logs
python src/train_drawdown_risk_models.py --verbose

# 7. Plot results
python src/model_visualizations.py
```

## Environment

Core dependencies:

- `python`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `requests`
- `yfinance`
- `pyarrow`

Install manually if needed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn requests yfinance pyarrow
```

## Caveats

- Universe membership is scraped from current index constituents, so historical survivorship bias can remain.
- Short-interest coverage is US-centric because the current source is FINRA.
- Fundamentals coverage is currently sparse and may limit the contribution of that feature block.
- Results artifacts in `results/stage1/` reflect the latest local pipeline run.

## Project Status

Current state:

- End-to-end pipeline is implemented and runnable
- Core evaluation outputs and visualizations are in place
- README and contributor-facing docs are now included

Next improvements:

- Add automated tests for feature engineering and split logic
- Add linting and formatting configuration
- Add experiment tracking or model registry support
- Add CI for smoke checks on README commands and script imports

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Author

**Jai Sharma**
