"""Static repository checks; intentionally avoids loading model libraries or data."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import store_sales_forecasting as pipeline  # noqa: E402


def main() -> int:
    module_path = ROOT / "store_sales_forecasting.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    required_functions = {
        "load_data", "merge_external_data", "prepare_datasets", "train_models",
        "predict_ensemble", "rmsle", "main",
    }
    missing = required_functions - functions
    assert not missing, f"missing public functions: {sorted(missing)}"
    compile(module_path.read_text(encoding="utf-8"), str(module_path), "exec")

    notebook_path = next(ROOT.glob("*.ipynb"))
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] >= 4
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    notebook_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "D:\\Data" not in notebook_source
    assert "LabelEncoder" not in notebook_source

    requirements = [line.split(";", 1)[0].split(">=", 1)[0].split("<", 1)[0].strip()
                    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
    for package in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost"):
        assert package in requirements, f"missing dependency: {package}"
    print("Static validation passed")
    return 0


def _write_fixture_data(root: Path, *, duplicate_oil: bool = False) -> None:
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    train_dates, test_dates = dates[:18], dates[18:]
    rows = []
    for date in dates:
        for store in (1, 2):
            for family in ("A", "B"):
                rows.append({
                    "id": len(rows) + 1,
                    "date": date,
                    "store_nbr": store,
                    "family": family,
                    "sales": float(store + (date.day % 4) + (family == "B")),
                    "onpromotion": float(date.day % 2),
                })
    all_rows = pd.DataFrame(rows)
    all_rows[all_rows.date.isin(train_dates)].to_csv(root / "train.csv", index=False)
    all_rows[all_rows.date.isin(test_dates)].drop(columns="sales").to_csv(root / "test.csv", index=False)
    pd.DataFrame([
        {"store_nbr": 1, "city": "Quito", "state": "Pichincha", "type": "A", "cluster": 1},
        {"store_nbr": 2, "city": "Cuenca", "state": "Azuay", "type": "B", "cluster": 2},
    ]).to_csv(root / "stores.csv", index=False)
    oil = pd.DataFrame({"date": dates, "dcoilwtico": np.linspace(70, 72, len(dates))})
    if duplicate_oil:
        oil = pd.concat([oil, oil.iloc[[0]]], ignore_index=True)
    oil.to_csv(root / "oil.csv", index=False)
    pd.DataFrame({
        "date": [dates[3], dates[10]], "type": ["Holiday", "Event"],
        "transferred": [False, False],
    }).to_csv(root / "holidays_events.csv", index=False)
    tx_rows = [{"date": date, "store_nbr": store, "transactions": 100 + store}
               for date in dates for store in (1, 2)]
    pd.DataFrame(tx_rows).to_csv(root / "transactions.csv", index=False)


def test_prepare_datasets_has_aligned_matrices_and_unique_date_holdout(tmp_path: Path) -> None:
    _write_fixture_data(tmp_path)
    datasets = pipeline.prepare_datasets(tmp_path)
    assert datasets["X_train"].shape[1] == datasets["X_val"].shape[1] == datasets["X_test"].shape[1]
    assert len(datasets["y_train"]) == len(datasets["X_train"])
    assert datasets["cutoff"] == pd.Timestamp("2024-01-18")
    assert datasets["X_val"].index.equals(pd.RangeIndex(len(datasets["X_val"])))
    assert np.isfinite(datasets["X_train"].to_numpy(dtype=float)).all()


def test_sales_history_masks_requested_rows() -> None:
    history = pd.DataFrame({
        "store_nbr": [1], "family": ["A"], "date": [pd.Timestamp("2024-01-01")], "sales": [10.0]
    })
    frame = pd.DataFrame({
        "store_nbr": [1, 1], "family": ["A", "A"],
        "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
    })
    features = pipeline._add_sales_history_features(history, frame)
    assert features.loc[0, "sales_lag_1"] == 10.0
    assert pd.isna(features.loc[1, "sales_lag_1"])


def test_load_data_rejects_duplicate_oil_dates(tmp_path: Path) -> None:
    _write_fixture_data(tmp_path, duplicate_oil=True)
    try:
        pipeline.load_data(tmp_path)
    except ValueError as error:
        assert "oil.csv" in str(error) and "one row per date" in str(error)
    else:
        raise AssertionError("duplicate oil dates were accepted")


def test_metric_and_ensemble_contracts() -> None:
    assert pipeline.rmsle(np.array([0.0, 2.0]), np.array([-1.0, 2.0])) >= 0
    try:
        pipeline.rmsle(np.array([-1.0]), np.array([0.0]))
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative actuals were accepted")

    class DummyModel:
        def __init__(self, values: list[float]):
            self.values = np.asarray(values, dtype=float)

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return self.values

    X = pd.DataFrame({"feature": [1.0, 2.0]})
    models = {name: DummyModel([0.0, 1.0]) for name in pipeline.ENSEMBLE_WEIGHTS}
    prediction = pipeline.predict_ensemble(models, X)
    assert prediction.shape == (2,) and np.isfinite(prediction).all() and (prediction >= 0).all()


if __name__ == "__main__":
    raise SystemExit(main())
