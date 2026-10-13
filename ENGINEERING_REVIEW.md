# Engineering Review — Ensemble-Learning-Approach-for-Retail-Sales-Prediction

**Repository:** `Abdulrahman181/Ensemble-Learning-Approach-for-Retail-Sales-Prediction`  
**Review scope:** Python, notebook, README, dependency declaration, generated-output hygiene, data splitting/leakage, reproducibility, and static validation.  
**Review status:** Repaired and statically validated; no end-to-end model run claimed.

## Executive summary

The repository originally contained only a README, an unstandardized `Requirements` file, a single notebook, and a pre-existing `submission_final.csv`. The notebook was not portable or safely reproducible: it used a hard-coded Windows path, had no missing-data contract, created train-only target-lag features while silently filling test lag columns with zeros, encoded categories with a `LabelEncoder` that can fail on unseen test values, shifted oil values after row expansion, split using a year quantile rather than a date boundary, retained large stale outputs, and did not control thread-level nondeterminism.

The repaired repository now has a reusable `store_sales_forecasting.py` module, a clean four-cell notebook, a conventional `requirements.txt`, `.gitignore`, and `tests/test_static_validation.py`. The pipeline fails clearly when the six Kaggle CSVs are unavailable and does not fabricate metrics or a submission.

## Findings and remediation

### Critical / high-impact findings

1. **Non-portable input path (fixed).** The original notebook read from `D:\Data\New folder9999999999\...`, which cannot work outside the author's machine. Data now resolve through `--data-dir` or `STORE_SALES_DATA_DIR`, with a documented default and an actionable `FileNotFoundError` listing missing files.
2. **Missing data and dependency contract (fixed).** The repository has no Kaggle input data. The module validates all six required files and required train/test columns before running. Heavy model imports are lazy, so the missing-data message is still available when optional dependencies are not installed.
3. **Feature leakage and train/test mismatch (fixed).** The original code created sales lags/rollups for train only, then added missing test columns as zero. The new feature builder tags requested rows explicitly, uses only prior history, masks validation/test sales before shifting, and retains the same feature schema for all matrices.
4. **Potential category failure (fixed).** `LabelEncoder` was fit separately per column and then called on test values; an unseen category raises an exception. `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` is now fitted on training rows and reused for validation/test.
5. **Incorrect row-grain shifts (fixed).** Oil lags were shifted after the oil table had effectively been joined to row-expanded sales data, so they were not reliably daily lags. Oil lag/rolling features are calculated once at the daily oil grain. Transaction lag/rolling features are calculated by store and date before merging.
6. **Validation leakage risk (fixed).** The original holdout rows could receive target history from other holdout rows because all train sales were shifted before the date split. The repaired chronological holdout masks holdout targets while deriving holdout target-history features. This is conservative and avoids reporting an optimistic score.

### Medium-impact findings

- The original notebook used `fillna(0)` across every column, conflating missing categories and numerical missingness. The repaired pipeline fills categories with `Unknown`, numerical features with zero after finite-value handling, and keeps target validation explicit.
- The original notebook did not validate merge cardinality. Store, oil, holiday, and transaction merges now use `validate="many_to_one"` where appropriate.
- The original workflow used uncontrolled multithreading and only partial random-state controls. Models now use `random_state=42`/`random_seed=42`, one thread, and deterministic LightGBM settings where supported.
- The original `Requirements` filename was nonstandard and lacked notebook support. It was replaced by `requirements.txt`; the dependency set includes the libraries actually imported by the repaired code and notebook.
- The original notebook contained stale execution counts, charts, model outputs, and hard-coded historical row counts. It was replaced with an output-free orchestration notebook.
- The original README claimed strong/stable results without a verifiable current run and referred to a notebook filename that did not exist. The README now distinguishes local holdout RMSLE from Kaggle leaderboard results and makes no performance claim.

## Current pipeline design

1. Strictly load and validate the six source tables.
2. Merge stores, daily oil features, cleaned holiday dates, and store-level transaction history with merge-cardinality checks.
3. Add calendar and promotion features.
4. Create sales-history features from prior observations only. Validation/test targets are masked during feature construction.
5. Use the latest 5% of unique chronological observations as a date-based holdout boundary, rejecting an empty partition.
6. Fit category encoding on training rows, use `log1p(sales)` as the model target, and train LightGBM, XGBoost, and CatBoost with fixed seeds and one thread.
7. Inverse-transform predictions, clip negatives to zero, calculate RMSLE, and write an `id,sales` submission only after a successful data-backed run.
8. Combine model predictions with the documented weights: XGBoost 0.50, LightGBM 0.40, CatBoost 0.10. These are configuration values, not a newly measured claim.

## Files changed or added

- `store_sales_forecasting.py` — new reusable CLI/module.
- `STORE SALES TIME SERIES FORECASTING .ipynb` — rewritten portable notebook; all stale outputs removed.
- `README.md` — corrected setup, data contract, leakage/reproducibility notes, and limitations.
- `requirements.txt` — conventional dependency declaration.
- `.gitignore` — ignores caches, checkpoints, and generated submissions.
- `tests/test_static_validation.py` — dependency-light AST/notebook/dependency checks.
- `ENGINEERING_REVIEW.md` — this report.
- `Requirements` — removed obsolete duplicate dependency filename.

The pre-existing `submission_final.csv` was not regenerated, validated, or represented as a new result. Its SHA-256 before the local commit is recorded only as an artifact identity: `2ee391161c93b8b5e3d86781bd95771c93a893891db65aef491dbd4d4f4ec3c63`.

## Validation performed

All checks below completed in the sandbox:

```text
python3 tests/test_static_validation.py
Static validation passed

python3 -m py_compile store_sales_forecasting.py
Passed

ruff check store_sales_forecasting.py tests/test_static_validation.py
All checks passed!

Notebook JSON parse and hygiene check
4 cells; 0 code-cell outputs; all code execution counts are null

Missing-data CLI check
Exited 2 with an actionable message listing train.csv, test.csv, stores.csv,
oil.csv, holidays_events.csv, and transactions.csv; no output was created.
```

## Limitations and next steps

- The six Kaggle data files are absent from the repository, so no model fitting, validation score, prediction distribution, or submission correctness can be claimed in this review.
- The current holdout strategy is chronological and leakage-aware, but target-lag features for later holdout/test rows become unavailable after masked rows; production forecasting may benefit from a documented recursive prediction strategy or a model that excludes target-derived features.
- Once data and dependencies are provisioned, run the CLI, inspect local per-model and ensemble RMSLE, check prediction finiteness/non-negativity, and compare the generated submission row count and IDs with `test.csv` before any external upload.
- No changes were pushed to GitHub. A local-only commit was created as requested.
