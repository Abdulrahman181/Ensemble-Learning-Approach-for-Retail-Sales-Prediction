# Store Sales Forecasting using Ensemble Learning

## 📌 Project Overview
This project focuses on predicting daily product sales for retail stores
using historical time-series data combined with external information
such as oil prices, holidays, store metadata, and transactions.

The solution uses advanced feature engineering techniques and
ensemble learning to improve forecasting accuracy.

---

## 📊 Dataset
The dataset is provided by Kaggle as part of the
**Store Sales - Time Series Forecasting** competition.

Main files used:
- train.csv
- test.csv
- stores.csv
- oil.csv
- holidays_events.csv
- transactions.csv

---

## ⚙️ Feature Engineering
Key engineered features include:
- Date-based features (year, month, day, day of week)
- Lag features for historical sales
- Rolling window statistics
- Promotion indicators
- Holiday and oil price effects

---

## 🤖 Models Used
- LightGBM
- XGBoost
- CatBoost

Each model is trained separately and evaluated using RMSLE.

---

## 🔗 Ensemble Strategy
Final predictions are generated using a weighted ensemble:
- 50% XGBoost
- 40% LightGBM
- 10% CatBoost

---

## 📈 Evaluation Metric
- Root Mean Squared Logarithmic Error (RMSLE)

---

## 📁 Files Included
- `store_sales_ensemble.ipynb` – Full training and prediction pipeline
- `submission_final.csv` – Final predictions file
- `README.md` – Project documentation

---

## 🏁 Results
The ensemble model achieves strong validation performance
and produces stable predictions for unseen test data.

---

## 🚀 Author
**Abdulrahman Ahmed**  
Artificial Intelligence Engineer
