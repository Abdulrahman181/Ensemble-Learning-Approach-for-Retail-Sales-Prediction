# Store Sales Forecasting with Ensemble Learning

This repository contains a portable forecasting pipeline for Kaggle's **Store Sales - Time Series Forecasting** data. The implementation is in [`store_sales_forecasting.py`](store_sales_forecasting.py); the notebook is a small interface to that module.

## Data requirements

The repository does **not** include the Kaggle input data. Put these files in one directory:

- `train.csv` (`id`, `date`, `store_nbr`, `family`, `sales`, `onpromotion`)
- `test.csv` (`id`, `date`, `store_nbr`, `family`, `onpromotion`)
- `stores.csv`
- `oil.csv`
- `holidays_events.csv`
- `transactions.csv`

Set `STORE_SALES_DATA_DIR` or pass `--data-dir`. If files are absent or incomplete, the CLI raises a clear error and does not create a fabricated submission.

## Setup and usage

```bash
python -m pip install -r requirements.txt
export STORE_SALES_DATA_DIR=/absolute/path/to/kaggle/store-sales-data
python store_sales_forecasting.py --data-dir "$STORE_SALES_DATA_DIR" --output submission_final.csv
```

Alternatively, open `STORE SALES TIME SERIES FORECASTING .ipynb`. The notebook skips training with an explanatory message when the data directory is unavailable.

## Engineering notes

- Inputs are loaded with portable `pathlib` paths and strict date parsing; the original machine-specific Windows path was removed.
- External merges use cardinality validation. Oil and transaction lag features are calculated at their natural date/store grain rather than by shifting a row-expanded table.
- The holdout is the latest 5% of **unique** training dates, with at least one complete date reserved when the dataset is large enough. Target-history features for validation and test rows are built from prior training history with holdout/test sales masked, preventing future target leakage. Categorical encoding is fitted on training rows and handles unknown categories as `-1`.
- Models learn `log1p(sales)` and predictions are inverse-transformed and clipped at zero before RMSLE. LightGBM, XGBoost, and CatBoost use fixed seeds and single-thread settings for more stable results. The documented ensemble weights are XGBoost 50%, LightGBM 40%, and CatBoost 10%.
- Source schemas, identifiers, merge grains, numeric domains, chronological train/test ordering, finite predictions, and output directory creation are validated before a submission is written. The printed RMSLE is a local chronological holdout score, **not** a Kaggle leaderboard result. No complete training run was claimed in this review because the input data are not present in the repository.

## Repository checks

Run the lightweight static checks without model training:

```bash
python tests/test_static_validation.py
pytest -q
python -m py_compile store_sales_forecasting.py
```

`submission_final.csv` is a pre-existing repository artifact and is not treated as a newly verified result by this review.
