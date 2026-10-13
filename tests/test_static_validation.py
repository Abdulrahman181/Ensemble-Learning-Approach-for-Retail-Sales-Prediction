"""Static repository checks; intentionally avoids loading model libraries or data."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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

    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    for package in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost"):
        assert package in requirements, f"missing dependency: {package}"
    print("Static validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
