"""Portable Store Sales forecasting pipeline.

The module intentionally does not run on import.  Use the CLI or call the public
functions from a notebook.  The Kaggle data files are expected in a directory
provided by ``--data-dir`` or ``STORE_SALES_DATA_DIR``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RANDOM_STATE = 42
ENSEMBLE_WEIGHTS = {"xgboost": 0.5, "lightgbm": 0.4, "catboost": 0.1}
REQUIRED_FILES = (
    "train.csv",
    "test.csv",
    "stores.csv",
    "oil.csv",
    "holidays_events.csv",
    "transactions.csv",
)
CATEGORICAL_COLUMNS = ["family", "city", "state", "type", "cluster"]
TABLE_REQUIRED_COLUMNS = {
    "stores": {"store_nbr", "city", "state", "type", "cluster"},
    "oil": {"date", "dcoilwtico"},
    "holidays": {"date", "type", "transferred"},
    "transactions": {"date", "store_nbr", "transactions"},
}


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Resolve and validate the input directory without assuming a local OS path."""
    candidate = data_dir or os.environ.get("STORE_SALES_DATA_DIR")
    if not candidate:
        candidate = Path(__file__).resolve().parent / "data"
    path = Path(candidate).expanduser().resolve()
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(
            f"Store Sales data is incomplete at {path}. Missing: {names}. "
            "Download the Kaggle Store Sales - Time Series Forecasting files "
            "and pass their directory with --data-dir or "
            "STORE_SALES_DATA_DIR."
        )
    return path


def _read_csv(path: Path, *, date_column: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if date_column:
        if date_column not in frame.columns:
            raise ValueError(f"{path.name} must contain a '{date_column}' column")
        parsed = pd.to_datetime(frame[date_column], errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"{path.name} contains missing or invalid {date_column} values")
        frame[date_column] = parsed
    return frame


def _require_columns(frame: pd.DataFrame, name: str, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name}.csv is missing required columns: {', '.join(missing)}")


def _coerce_numeric(frame: pd.DataFrame, column: str, name: str) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid = values.isna() & frame[column].notna()
    if invalid.any():
        raise ValueError(f"{name}.csv column '{column}' must be numeric")
    frame[column] = values


def _require_integer(frame: pd.DataFrame, column: str, name: str) -> None:
    values = frame[column]
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{name}.csv column '{column}' must contain finite integers")


def load_data(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load the six Kaggle tables and parse dates strictly."""
    root = resolve_data_dir(data_dir)
    data = {
        "train": _read_csv(root / "train.csv", date_column="date"),
        "test": _read_csv(root / "test.csv", date_column="date"),
        "stores": _read_csv(root / "stores.csv"),
        "oil": _read_csv(root / "oil.csv", date_column="date"),
        "holidays": _read_csv(root / "holidays_events.csv", date_column="date"),
        "transactions": _read_csv(root / "transactions.csv", date_column="date"),
    }
    required_train = {"id", "date", "store_nbr", "family", "sales", "onpromotion"}
    required_test = {"id", "date", "store_nbr", "family", "onpromotion"}
    _require_columns(data["train"], "train", required_train)
    _require_columns(data["test"], "test", required_test)
    for name, required in TABLE_REQUIRED_COLUMNS.items():
        _require_columns(data[name], name, required)
    for name in ("train", "test", "stores", "oil", "transactions"):
        if data[name].empty:
            raise ValueError(f"{name}.csv must not be empty")
    for name in ("train", "test"):
        frame = data[name]
        for column in ("id", "store_nbr", "onpromotion"):
            _coerce_numeric(frame, column, name)
        _require_integer(frame, "id", name)
        _require_integer(frame, "store_nbr", name)
        if frame[["id", "date", "store_nbr", "family"]].isna().any().any():
            raise ValueError(f"{name}.csv contains null identifier, date, store, or family values")
        if frame["id"].duplicated().any():
            raise ValueError(f"{name}.csv must contain unique id values")
        if frame[["date", "store_nbr", "family"]].duplicated().any():
            raise ValueError(f"{name}.csv must contain unique date/store_nbr/family rows")
        if (frame["onpromotion"].isna().any()
                or not np.isfinite(frame["onpromotion"]).all()
                or (frame["onpromotion"] < 0).any()):
            raise ValueError(f"{name}.csv onpromotion must be finite, non-null, and non-negative")
    _coerce_numeric(data["train"], "sales", "train")
    if data["train"]["sales"].isna().any() or not np.isfinite(data["train"]["sales"]).all() or (data["train"]["sales"] < 0).any():
        raise ValueError("train.csv sales must be non-null and non-negative for RMSLE training")
    if data["stores"]["store_nbr"].duplicated().any():
        raise ValueError("stores.csv must contain one row per store_nbr")
    _coerce_numeric(data["stores"], "store_nbr", "stores")
    _require_integer(data["stores"], "store_nbr", "stores")
    _coerce_numeric(data["transactions"], "store_nbr", "transactions")
    _require_integer(data["transactions"], "store_nbr", "transactions")
    if data["oil"]["date"].duplicated().any():
        raise ValueError("oil.csv must contain one row per date")
    if data["transactions"][["date", "store_nbr"]].duplicated().any():
        raise ValueError("transactions.csv must contain one row per date/store_nbr")
    _coerce_numeric(data["oil"], "dcoilwtico", "oil")
    _coerce_numeric(data["transactions"], "transactions", "transactions")
    known_stores = set(data["stores"]["store_nbr"])
    for name in ("train", "test"):
        unknown_stores = sorted(set(data[name]["store_nbr"]) - known_stores)
        if unknown_stores:
            raise ValueError(f"{name}.csv references stores absent from stores.csv: {unknown_stores}")
    return data


def _build_oil_features(oil: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "dcoilwtico"}
    missing = required.difference(oil.columns)
    if missing:
        raise ValueError(f"oil.csv is missing required columns: {', '.join(sorted(missing))}")
    daily = oil[["date", "dcoilwtico"]].sort_values("date").copy()
    daily["dcoilwtico"] = pd.to_numeric(daily["dcoilwtico"], errors="coerce")
    if daily["dcoilwtico"].notna().any() and not np.isfinite(daily["dcoilwtico"].dropna()).all():
        raise ValueError("oil.csv dcoilwtico must contain finite numeric values")
    # Forward-fill only: back-filling would use a future oil observation for an
    # early missing date and can leak future information into a time split.
    daily["dcoilwtico"] = daily["dcoilwtico"].ffill()
    for lag in (1, 7):
        daily[f"oil_lag_{lag}"] = daily["dcoilwtico"].shift(lag)
    daily["oil_roll7"] = daily["dcoilwtico"].shift(1).rolling(7, min_periods=1).mean()
    return daily


def _build_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "store_nbr", "transactions"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(
            f"transactions.csv is missing required columns: {', '.join(sorted(missing))}"
        )
    tx = transactions[["date", "store_nbr", "transactions"]].copy()
    tx["transactions"] = pd.to_numeric(tx["transactions"], errors="coerce")
    if tx["transactions"].notna().any() and not np.isfinite(tx["transactions"].dropna()).all():
        raise ValueError("transactions.csv transactions must contain finite numeric values")
    if tx[["date", "store_nbr"]].duplicated().any():
        raise ValueError("transactions.csv must contain one row per date/store_nbr")
    tx = tx.sort_values(["store_nbr", "date"])
    tx["transactions_lag1"] = tx.groupby("store_nbr")["transactions"].shift(1)
    tx["transactions_roll7"] = tx.groupby("store_nbr")["transactions"].transform(
        lambda values: values.shift(1).rolling(7, min_periods=1).mean()
    )
    return tx


def merge_external_data(
    frame: pd.DataFrame,
    stores: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Merge exogenous tables with cardinality checks and no target columns."""
    _require_columns(frame, "input", {"date", "store_nbr", "family", "onpromotion"})
    _require_columns(stores, "stores", TABLE_REQUIRED_COLUMNS["stores"])
    _require_columns(holidays, "holidays_events", TABLE_REQUIRED_COLUMNS["holidays"])
    result = frame.copy()
    result = result.merge(stores, on="store_nbr", how="left", validate="many_to_one")
    result = result.merge(_build_oil_features(oil), on="date", how="left", validate="many_to_one")

    transferred = holidays["transferred"].astype(str).str.strip().str.lower().eq("true")
    holiday_dates = holidays.loc[
        (~transferred) & (holidays["type"] != "Transfer"), ["date"]
    ].drop_duplicates()
    holiday_dates["is_holiday"] = 1
    result = result.merge(holiday_dates, on="date", how="left", validate="many_to_one")

    result = result.merge(
        _build_transaction_features(transactions),
        on=["date", "store_nbr"],
        how="left",
        validate="many_to_one",
    )
    for column in ("is_holiday", "transactions", "transactions_lag1", "transactions_roll7"):
        if column in result:
            result[column] = result[column].fillna(0)
    result["is_holiday"] = result["is_holiday"].astype("int8")
    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month
    result["day"] = result["date"].dt.day
    result["dayofweek"] = result["date"].dt.dayofweek
    result["is_weekend"] = result["dayofweek"].isin((5, 6)).astype("int8")
    result["is_promotion"] = (result["onpromotion"].fillna(0) > 0).astype("int8")
    return result


def _add_sales_history_features(history: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Add target lags using only rows in ``history`` before each frame row.

    For validation/test rows, their sales are masked before shifting.  Thus a
    holdout row cannot use another holdout row's observed target as a feature.
    """
    keys = ["store_nbr", "family"]
    history_cols = keys + ["date", "sales"]
    history_part = history[history_cols].assign(_is_frame=0, _frame_order=-1)
    frame_part = frame[keys + ["date"]].assign(
        sales=np.nan, _is_frame=1, _frame_order=np.arange(len(frame))
    )
    combined = pd.concat([history_part, frame_part], ignore_index=True).sort_values(
        keys + ["date", "_is_frame"], kind="mergesort"
    )
    grouped = combined.groupby(keys, sort=False)["sales"]
    for lag in (1, 7, 14):
        combined[f"sales_lag_{lag}"] = grouped.shift(lag)
    for window in (7, 30):
        combined[f"sales_roll_mean_{window}"] = grouped.transform(
            lambda values, window=window: values.shift(1).rolling(window, min_periods=1).mean()
        )
    feature_names = [
        "sales_lag_1", "sales_lag_7", "sales_lag_14",
        "sales_roll_mean_7", "sales_roll_mean_30",
    ]
    generated = (
        combined.loc[combined["_is_frame"].eq(1)]
        .sort_values("_frame_order")[feature_names]
        .reset_index(drop=True)
    )
    result = frame.reset_index(drop=True).copy()
    for name in feature_names:
        result[name] = generated[name].to_numpy()
    return result


def _fill_and_encode(train: pd.DataFrame, *others: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    from sklearn.preprocessing import OrdinalEncoder

    train = train.copy()
    transformed = [other.copy() for other in others]
    for column in CATEGORICAL_COLUMNS:
        if column not in train or any(column not in other for other in transformed):
            raise ValueError(f"Expected categorical column '{column}' after store merge")
        train[column] = train[column].fillna("Unknown").astype(str)
        for other in transformed:
            other[column] = other[column].fillna("Unknown").astype(str)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train[CATEGORICAL_COLUMNS] = encoder.fit_transform(train[CATEGORICAL_COLUMNS])
    for other in transformed:
        other[CATEGORICAL_COLUMNS] = encoder.transform(other[CATEGORICAL_COLUMNS])
    for column in train.columns:
        if column in ("date", "sales"):
            continue
        if pd.api.types.is_numeric_dtype(train[column]):
            train[column] = train[column].replace([np.inf, -np.inf], np.nan).fillna(0)
            for other in transformed:
                other[column] = other[column].replace([np.inf, -np.inf], np.nan).fillna(0)
    return (train, *transformed)


def prepare_datasets(data_dir: str | Path | None = None) -> dict[str, Any]:
    """Prepare leakage-aware train/validation/test matrices."""
    data = load_data(data_dir)
    train = merge_external_data(data["train"], data["stores"], data["oil"], data["holidays"], data["transactions"])
    test = merge_external_data(data["test"], data["stores"], data["oil"], data["holidays"], data["transactions"])
    train = train.sort_values("date").reset_index(drop=True)
    test = test.sort_values("date").reset_index(drop=True)
    if test["date"].min() <= train["date"].max():
        raise ValueError("test.csv dates must be strictly later than all train.csv dates")
    unique_dates = train["date"].drop_duplicates().sort_values().reset_index(drop=True)
    validation_count = max(1, int(np.ceil(len(unique_dates) * 0.05)))
    if validation_count >= len(unique_dates):
        raise ValueError("The date split needs at least two distinct training dates")
    cutoff = unique_dates.iloc[-validation_count]
    fit_mask = train["date"] < cutoff
    if not fit_mask.any() or fit_mask.all():
        raise ValueError("The date split produced an empty training or validation partition")
    fit = train.loc[fit_mask].copy()
    validation = train.loc[~fit_mask].copy()
    fit = _add_sales_history_features(fit, fit)
    validation = _add_sales_history_features(fit, validation)
    test = _add_sales_history_features(train, test)

    drop = ["id", "date", "sales"]
    target = fit["sales"].copy()
    validation_target = validation["sales"].copy()
    fit, validation, test = _fill_and_encode(fit, validation, test)
    features = [column for column in fit.columns if column not in drop]
    return {
        "X_train": fit[features],
        "y_train": np.log1p(target.to_numpy()),
        "X_val": validation[features],
        "y_val": validation_target.to_numpy(),
        "X_test": test[features],
        "test_id": test["id"].reset_index(drop=True),
        "features": features,
        "cutoff": cutoff,
    }


def rmsle(y_true: np.ndarray, predictions: np.ndarray) -> float:
    """Compute RMSLE after clipping negative forecasts to the metric domain."""
    from sklearn.metrics import mean_squared_log_error

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("RMSLE inputs must have the same non-empty shape")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("RMSLE inputs must contain only finite values")
    if (actual < 0).any():
        raise ValueError("RMSLE actual values must be non-negative")
    predicted = np.clip(predicted, 0, None)
    return float(np.sqrt(mean_squared_log_error(actual, predicted)))


def train_models(datasets: dict[str, Any]) -> dict[str, Any]:
    """Train the three configured regressors deterministically on log-sales."""
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    np.random.seed(RANDOM_STATE)
    X_train, y_train = datasets["X_train"], datasets["y_train"]
    models = {
        "lightgbm": LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
            n_jobs=1, deterministic=True, verbosity=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
            tree_method="hist", n_jobs=1, verbosity=0,
        ),
        "catboost": CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=8,
            random_seed=RANDOM_STATE, thread_count=1, verbose=False,
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def predict_ensemble(models: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Predict on the original sales scale using the documented weights."""
    missing = sorted(set(ENSEMBLE_WEIGHTS).difference(models))
    if missing:
        raise KeyError(f"Missing ensemble model(s): {', '.join(missing)}")
    if not np.isclose(sum(ENSEMBLE_WEIGHTS.values()), 1.0):
        raise ValueError("ENSEMBLE_WEIGHTS must sum to 1")
    total = np.zeros(len(X), dtype=float)
    for name, weight in ENSEMBLE_WEIGHTS.items():
        log_prediction = np.asarray(models[name].predict(X), dtype=float)
        if log_prediction.shape != (len(X),) or not np.isfinite(log_prediction).all():
            raise ValueError(f"{name} produced invalid log-scale predictions")
        # Guard the inverse transform against an accidental model overflow;
        # normal log1p(sales) predictions are far below this bound.
        safe_log = np.clip(log_prediction, -745.0, np.log(np.finfo(float).max) - 1.0)
        original_scale = np.expm1(safe_log)
        if not np.isfinite(original_scale).all():
            raise ValueError(f"{name} produced non-finite sales predictions")
        total += weight * original_scale
    if not np.isfinite(total).all():
        raise ValueError("Ensemble produced non-finite sales predictions")
    return np.clip(total, 0, None)


def run(data_dir: str | Path | None, output: str | Path) -> dict[str, float]:
    """Train, report holdout RMSLE, and write a submission without fabricating results."""
    datasets = prepare_datasets(data_dir)
    models = train_models(datasets)
    scores = {}
    for name, model in models.items():
        log_prediction = np.asarray(model.predict(datasets["X_val"]), dtype=float)
        if not np.isfinite(log_prediction).all():
            raise ValueError(f"{name} produced non-finite validation predictions")
        scores[name] = rmsle(datasets["y_val"], np.expm1(np.clip(log_prediction, -745.0, 700.0)))
    predictions = predict_ensemble(models, datasets["X_test"])
    scores["ensemble"] = rmsle(
        datasets["y_val"], predict_ensemble(models, datasets["X_val"])
    )
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": datasets["test_id"], "sales": predictions}).to_csv(output_path, index=False)
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None, help="Directory containing the six Kaggle CSV files")
    parser.add_argument("--output", default="submission_final.csv", help="Output submission CSV path")
    args = parser.parse_args()
    try:
        scores = run(args.data_dir, args.output)
    except (FileNotFoundError, ValueError, KeyError, OSError, ImportError) as exc:
        parser.error(str(exc))
    print("Holdout RMSLE (not Kaggle leaderboard scores):")
    for name, score in scores.items():
        print(f"  {name}: {score:.5f}")
    print(f"Submission written to {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
