# Store Sales Forecasting with Ensemble Learning

This repository contains a Jupyter notebook for experimenting with daily retail sales forecasting using the Kaggle **Store Sales - Time Series Forecasting** data. The notebook fits LightGBM, XGBoost, and CatBoost models and writes a weighted prediction file. It is an exploratory project, not a verified or reproducible benchmark.

## Data

The six competition files required by the notebook are not included in this repository:

- `train.csv`
- `test.csv`
- `stores.csv`
- `oil.csv`
- `holidays_events.csv`
- `transactions.csv`

Obtain the files from the [Kaggle competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) and follow its terms. Place them together in a local directory (the default is `data/`, which is intentionally not tracked), or set `STORE_SALES_DATA_DIR` to their directory. For example, in a POSIX shell:

```sh
export STORE_SALES_DATA_DIR="$HOME/datasets/store-sales"
```

On Windows PowerShell, set the variable with `$env:STORE_SALES_DATA_DIR = 'C:\path\to\store-sales'` for the current session. The notebook reports the missing filenames before loading if the expected files are not present.

## Setup and run

Use Python 3.10 or newer in a virtual environment, then install the notebook dependencies:

```sh
python -m venv .venv
# Activate .venv using the command for your operating system
python -m pip install -r requirements.txt jupyter
jupyter notebook "STORE SALES TIME SERIES FORECASTING .ipynb"
```

The dependency list is not version-locked: this repository has no recorded, verified environment, so the listed packages should not be interpreted as a tested compatibility matrix. Record and pin versions after validating the full notebook with the competition data if reproducible results are needed.

## What the notebook does

It merges store, oil, holiday, and transaction data; creates calendar and time-series features; trains the three regressors; compares individual validation RMSLE values; then combines test predictions with fixed weights (50% XGBoost, 40% LightGBM, 10% CatBoost) and writes `submission_final.csv` in the notebook's current working directory.

## Important limitations

- The raw data is absent, so the training and prediction workflow has not been rerun as part of this repository repair. The checked-in `submission_final.csv` is a pre-existing artifact; it has not been verified against the competition test IDs or regenerated from the notebook.
- In the current notebook, lag and rolling features are calculated on `train` only. At prediction time, missing test feature columns are filled with zero. Transactions are also merged directly by date and store, with missing test values filled by zero. Consequently, the test-time features do not match a demonstrated, leakage-safe forecasting setup. The saved predictions and validation numbers must not be treated as verified evidence of generalization.
- Validation uses a year-quantile split and does not establish that the validation procedure reproduces the competition's multi-day forecast horizon. The ensemble weights are fixed, and no ensemble score is calculated in the notebook.
- The notebook saves the output CSV relative to the current working directory; check the output location before replacing any existing file.

These modeling limitations require an explicitly defined forecast horizon and access to the original data to repair and validate responsibly. No accuracy or model-performance claim is made here.

## Files

- `STORE SALES TIME SERIES FORECASTING .ipynb` — training and prediction notebook.
- `requirements.txt` — unpinned Python package names used by the notebook.
- `submission_final.csv` — checked-in prediction artifact of unverified provenance.
