# ChainSight-AI: Model Inspection Report

**Date of Audit:** November 19, 2026  
**Auditor:** Senior MLOps Engineer & Platform Architect  
**Project:** ChainSight-AI (Autonomous Supply Chain Resilience & Decision Intelligence Platform)  
**Status:** Phase 0 Completed — Ready for Verification  

---

## 1. Executive Summary & Audit Methodology

A comprehensive, non-destructive audit of all four trained machine learning models, their serialized artifacts (`.joblib`), metadata files (`.json`), training evaluation records (`.csv`, `.txt`), visualizations, and datasets was conducted across both repository locations:
- `models/` (application inference artifacts)
- `chainsight_ai_complete_models/` (training outputs, performance metrics, and complete model metadata)

Every model was inspected via Python deserialization, introspecting internal estimator architectures, feature names, categorical encodings, pipeline transformers, parameter configurations, and prediction behaviors. Sample inference was verified against real model signatures.

---

## 2. Model 1: Demand Forecasting

### 2.1 Basic Metadata
- **Model Name:** Demand Forecasting Model (`demand_forecasting`)
- **Joblib Artifact Locations:**
  - `models/demand_forecasting/demand_model.joblib`
  - `chainsight_ai_complete_models/demand_forecasting/demand_model.joblib`
- **Python Object Type:** `lightgbm.sklearn.LGBMRegressor`
- **Underlying Engine:** LightGBM Booster (`gbdt`)
- **Target Variable:** Target product unit sales for next 7 days (`target_definition: "Target sales for next 7 days"`)
- **Problem Formulation:** Time-series tabular regression

### 2.2 Input Schema & Feature Ordering
The model requires exactly **12 input features** in the following strict order:

| Index | Feature Name | Expected Data Type | Description / Domain |
|---|---|---|---|
| 0 | `wday` | Integer (1–7) | Day of week (M5 convention: 1 = Saturday, ..., 7 = Friday) |
| 1 | `month` | Integer (1–12) | Calendar month |
| 2 | `year` | Integer | Calendar year (e.g., 2011–2016) |
| 3 | `sell_price` | Float | Unit selling price of the product |
| 4 | `lag_7` | Float | Unit sales lag 7 days prior |
| 5 | `lag_14` | Float | Unit sales lag 14 days prior |
| 6 | `lag_28` | Float | Unit sales lag 28 days prior |
| 7 | `rolling_mean_7` | Float | 7-day rolling average unit sales |
| 8 | `rolling_mean_28`| Float | 28-day rolling average unit sales |
| 9 | `rolling_std_7` | Float | 7-day rolling standard deviation of unit sales |
| 10 | `dept_id` | Categorical / String | Department identifier |
| 11 | `cat_id` | Categorical / String | Product category identifier |

### 2.3 Preprocessing Requirements & Categorical Encoding
- **Preprocessing Included in Saved Artifact:** **No.** The saved artifact is a bare `LGBMRegressor` estimator, not an sklearn `Pipeline`.
- **Critical Technical Detail:** The underlying LightGBM booster was trained with native pandas categorical dtypes. The booster's `pandas_categorical` metadata explicitly registers:
  - `dept_id` categories: `['FOODS_1', 'FOODS_2', 'FOODS_3', 'HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2']`
  - `cat_id` categories: `['FOODS', 'HOBBIES', 'HOUSEHOLD']`
- **Inference Requirement:** The prediction service must convert incoming `dept_id` and `cat_id` values into `pd.Categorical(values, categories=...)` matching these exact registered categories prior to calling `model.predict()`. Otherwise, LightGBM raises `ValueError: train and valid dataset categorical_feature do not match.`

### 2.4 Hyperparameters
Recovered from `chainsight_ai_complete_models/demand_forecasting/training_metadata.json`:
- `boosting_type`: `gbdt`
- `learning_rate`: `0.05`
- `n_estimators`: `100`
- `num_leaves`: `31`
- `max_depth`: `-1` (unconstrained)
- `min_child_samples`: `20`
- `min_child_weight`: `0.001`
- `subsample`: `1.0`
- `colsample_bytree`: `1.0`
- `random_state`: `42`
- `importance_type`: `split`
- `verbose`: `-1`

### 2.5 Evaluation Metrics & Dataset Information
- **Dataset:** Walmart M5 Forecasting competition (`public_datasets/m5/` containing `calendar.csv`, `sales_train_validation.csv`, `sell_prices.csv`)
- **Training Date Range:** `2011-03-04` to `2016-03-26`
- **Reported Metrics (`metrics.json`):**
  - LightGBM MAE: `0.9324` (Baseline MAE: `1.1193`)
  - LightGBM RMSE: `1.7573` (Baseline RMSE: `2.2818`)
  - LightGBM WAPE: `0.7535` (Baseline WAPE: `0.9045`)
- **Inference Status:** Verified functional. Sample test inference returns `[13.3547]` units.

---

## 3. Model 2: Stockout Prediction

### 3.1 Basic Metadata
- **Model Name:** Stockout Prediction Model (`stockout_prediction`)
- **Joblib Artifact Locations:**
  - `models/stockout_prediction/stockout_model.joblib`
  - `chainsight_ai_complete_models/stockout_prediction/stockout_model.joblib`
- **Python Object Type:** `sklearn.pipeline.Pipeline`
- **Underlying Engine:** Scikit-learn Pipeline with `ColumnTransformer` + `lightgbm.sklearn.LGBMClassifier`
- **Target Variable:** Binary stockout within next 7 days (`stockout_next_7d`: 0 = No Stockout, 1 = Stockout)
- **Problem Formulation:** Binary classification with probabilistic scoring

### 3.2 Input Schema & Feature Ordering
The pipeline expects a pandas DataFrame with **13 features**:

| Index | Feature Name | Data Type | Role in Pipeline | Valid Domain / Categories |
|---|---|---|---|---|
| 0 | `opening_stock` | Integer / Float | Numerical (StandardScaler) | Non-negative inventory count |
| 1 | `received_stock` | Integer / Float | Numerical (StandardScaler) | Incoming stock units |
| 2 | `avg_daily_demand_7d` | Float | Numerical (StandardScaler) | Units/day over last 7 days |
| 3 | `avg_daily_demand_30d` | Float | Numerical (StandardScaler) | Units/day over last 30 days |
| 4 | `demand_std_7d` | Float | Numerical (StandardScaler) | Volatility of 7-day demand |
| 5 | `inventory_coverage_days` | Float | Numerical (StandardScaler) | Stock / daily demand ratio |
| 6 | `supplier_avg_lead_time` | Integer / Float | Numerical (StandardScaler) | Supplier lead time in days |
| 7 | `supplier_reliability` | Float | Numerical (StandardScaler) | Ratio 0.0 to 1.0 (on-time rate) |
| 8 | `warehouse_utilization_pct` | Float | Numerical (StandardScaler) | Warehouse fill percentage (0–100) |
| 9 | `warehouse_id` | String | Categorical (OneHotEncoder) | `['WH_001', 'WH_002', 'WH_003', 'WH_004', 'WH_005']` |
| 10 | `product_category` | String | Categorical (OneHotEncoder) | `['Automotive', 'Electronics', 'Health_Beauty', 'Home_Decor', 'Sports_Outdoors']` |
| 11 | `day_of_week` | Integer | Categorical (OneHotEncoder) | `0` to `6` (0 = Monday, ..., 6 = Sunday) |
| 12 | `month` | Integer | Categorical (OneHotEncoder) | `1` to `12` |

### 3.3 Preprocessing Requirements & Embedded Pipeline Structure
- **Preprocessing Included in Saved Artifact:** **Yes, 100% self-contained.**
- The artifact is an end-to-end `sklearn.pipeline.Pipeline` with two steps:
  1. `preprocessor`: `ColumnTransformer` containing:
     - `num`: `StandardScaler` fitted on the 9 numerical columns.
     - `cat`: `OneHotEncoder(handle_unknown='ignore')` fitted on the 4 categorical columns.
  2. `classifier`: `LGBMClassifier` (100 estimators, max depth 5, learning rate 0.08).
- **Critical Dependency Warning:** The artifact was serialized using `scikit-learn==1.6.1`. Attempting to load this artifact on `scikit-learn >= 1.8.0` raises `AttributeError: Can't get attribute '_RemainderColsList' on 'sklearn.compose._column_transformer'`. Strict pinning to `scikit-learn==1.6.1` is mandatory.

### 3.4 Hyperparameters & Optimal Decision Threshold
Recovered from `training_metadata.json` and `metrics.json`:
- `learning_rate`: `0.08`
- `max_depth`: `5`
- `n_estimators`: `100`
- `num_leaves`: `31`
- `random_state`: `42`
- **Selected F1 Decision Threshold:** `0.45560` (probability threshold calibrated for maximum F1 score)

### 3.5 Evaluation Metrics & Dataset Information
- **Dataset:** `stockout_training_data.csv` (358,500 rows, simulated supply chain inventory records)
- **Training Date Range:** `2021-01-07` to `2022-05-22` | **Testing Range:** `2022-09-08` to `2022-12-24`
- **Reported Metrics (`metrics.json` / `classification_report.txt`):**
  - Accuracy: `96.57%`
  - Precision: `92.20%`
  - Recall: `95.96%`
  - F1-Score: `0.9405`
  - ROC-AUC: `0.9901`
  - PR-AUC: `0.9600`
- **Inference Status:** Verified functional. Sample inference returns binary prediction `1` with class probabilities `[0.4560, 0.5440]`.

---

## 4. Model 3: Delivery ETA Prediction

### 4.1 Basic Metadata
- **Model Name:** Delivery ETA Prediction Model (`delivery_eta`)
- **Joblib Artifact Locations:**
  - `models/delivery_eta/delivery_eta_model.joblib`
  - `chainsight_ai_complete_models/delivery_eta/delivery_eta_model.joblib`
- **Python Object Type:** `catboost.core.CatBoostRegressor`
- **Underlying Engine:** CatBoost Regressor (gradient boosting on decision trees with native categorical handling)
- **Target Variable:** `actual_delivery_duration` (actual order transit duration in days)
- **Problem Formulation:** Tabular regression (days)

### 4.2 Input Schema & Feature Ordering
The model requires **7 features** in the following exact order:

| Index | Feature Name | Data Type | Category / Type | Example Value |
|---|---|---|---|---|
| 0 | `price` | Float | Numerical | `45.00` |
| 1 | `freight_value` | Float | Numerical | `12.50` |
| 2 | `product_weight_g` | Float | Numerical | `500.0` |
| 3 | `product_category_name` | String / Object | Categorical | `"beleza_saude"` |
| 4 | `seller_state` | String / Object | Categorical | `"SP"` |
| 5 | `customer_state` | String / Object | Categorical | `"RJ"` |
| 6 | `seller_delivery_count`| Integer / Float | Numerical | `15` |

### 4.3 Preprocessing Requirements
- **Preprocessing Included in Saved Artifact:** **Yes.** CatBoost embeds its own categorical feature mapping directly in the model binary (`cat_features=['product_category_name', 'seller_state', 'customer_state']`).
- No separate scaler or one-hot encoder is required. Strings can be passed directly in the input DataFrame.
- **Library Requirement:** `catboost>=1.2.0` must be installed.

### 4.4 Hyperparameters
Recovered via object introspection:
- `iterations`: `150`
- `learning_rate`: `0.1`
- `depth`: `6`
- `loss_function`: `RMSE`
- `random_seed`: `42`
- `verbose`: `0`

### 4.5 Evaluation Metrics & Dataset Information
- **Dataset:** Brazilian E-Commerce Public Dataset by Olist (`public_datasets/olist/`)
- **Reported Metrics (`metrics.json` & `all_models_comparison.csv`):**
  - CatBoost MAE: `6.1624` days (Baseline MAE: `6.7353` days)
  - CatBoost RMSE: `7.2760` days (Baseline RMSE: `7.7199` days)
  - $R^2$: `-0.7749` (indicating high variance and difficult tail distribution in transit times)
- **Inference Status:** Verified functional. Sample test inference returns `12.2612` days.

---

## 5. Model 4: Seller Delay Risk Prediction

### 5.1 Basic Metadata
- **Model Name:** Seller Delay Risk Prediction Model (`seller_risk`)
- **Joblib Artifact Locations:**
  - `models/seller_risk/seller_risk_model.joblib`
  - `chainsight_ai_complete_models/seller_risk/seller_risk_model.joblib`
- **Python Object Type:** `catboost.core.CatBoostClassifier`
- **Underlying Engine:** CatBoost Classifier
- **Target Variable:** `delayed` (binary: 0 = On Time, 1 = Delivery Delayed beyond estimated date)
- **Problem Formulation:** Imbalanced binary classification

### 5.2 Input Schema & Feature Ordering
The model requires the identical **7 features** as Delivery ETA:

| Index | Feature Name | Data Type | Category / Type | Example Value |
|---|---|---|---|---|
| 0 | `price` | Float | Numerical | `45.00` |
| 1 | `freight_value` | Float | Numerical | `12.50` |
| 2 | `product_weight_g` | Float | Numerical | `500.0` |
| 3 | `product_category_name` | String / Object | Categorical | `"beleza_saude"` |
| 4 | `seller_state` | String / Object | Categorical | `"SP"` |
| 5 | `customer_state` | String / Object | Categorical | `"RJ"` |
| 6 | `seller_delivery_count`| Integer / Float | Numerical | `15` |

### 5.3 Preprocessing Requirements
- **Preprocessing Included in Saved Artifact:** **Yes.** CatBoost manages categorical feature encoding internally (`cat_features=['product_category_name', 'seller_state', 'customer_state']`).
- `classes_`: `[0, 1]`. Supports `predict()` and `predict_proba()`.
- **Class Balancing:** `auto_class_weights='Balanced'` was applied during training due to severe class imbalance in shipping delays.

### 5.4 Hyperparameters
Recovered via object introspection:
- `iterations`: `150`
- `learning_rate`: `0.1`
- `depth`: `6`
- `auto_class_weights`: `Balanced`
- `random_seed`: `42`
- `verbose`: `0`

### 5.5 Evaluation Metrics & Dataset Information
- **Dataset:** Olist Brazilian E-Commerce dataset (`public_datasets/olist/`)
- **Reported Metrics (`metrics.json`):**
  - Precision: `0.0445` (4.45%)
  - Recall: `0.2236` (22.36%)
  - F1-Score: `0.0743`
  - ROC-AUC: `0.4113`
  - PR-AUC: `0.0539`
- **Operational Assessment:** This model has low discriminatory power on the raw Olist historical features, likely reflecting stochastic delivery delays in regional carrier networks rather than pure seller-level causation. The API response will transparently convey calibrated risk scores, risk levels (Low/Medium/High), and highlight these model limitations.
- **Inference Status:** Verified functional. Sample test inference returns prediction `0` (Not Delayed) with class probabilities `[0.5501, 0.4499]`.

---

## 6. Summary Comparison Matrix

| Model | Task | Algorithm | Input Features | Preprocessing Embedded? | Primary Metric | Inference Status |
|---|---|---|---|---|---|---|
| **Demand Forecasting** | Regression | LightGBM Regressor | 12 | No (Pandas Categorical needed) | MAE: 0.9324 | Verified (13.35 units) |
| **Stockout Prediction** | Classification | Scikit-Learn Pipeline (LGBM) | 13 | Yes (ColumnTransformer + Scaler + OHE) | F1: 0.9405, AUC: 0.9901 | Verified (Class 1, 54.4% prob) |
| **Delivery ETA** | Regression | CatBoost Regressor | 7 | Yes (Native CatBoost) | MAE: 6.16 days | Verified (12.26 days) |
| **Seller Risk** | Classification | CatBoost Classifier | 7 | Yes (Native CatBoost) | Recall: 0.2236, F1: 0.0743 | Verified (Class 0, 45.0% prob) |

---

## 7. Dependency & Compatibility Analysis

1. **`scikit-learn` Version Pinning:**
   - `stockout_model.joblib` was serialized with `scikit-learn==1.6.1`.
   - Running under `scikit-learn>=1.8.0` breaks deserialization of `ColumnTransformer` due to internal refactoring of `_RemainderColsList`.
   - **Resolution:** Pin `scikit-learn==1.6.1` in `requirements.txt` and Docker images.
2. **`catboost` Requirement:**
   - Both `delivery_eta` and `seller_risk` require `catboost>=1.2.0`.
   - **Resolution:** Add `catboost>=1.2.0` to `requirements.txt`.
3. **`lightgbm` Requirement:**
   - `demand_forecasting` and `stockout_prediction` require `lightgbm>=4.0.0`.
   - **Resolution:** Verified working with `lightgbm>=4.5.0`.
4. **`mlflow` Requirement:**
   - Needed for Phase 4 Model Registry and tracking. Pin `mlflow>=2.15.0`.
5. **Python Version Compatibility:**
   - Compatible with Python 3.10 and Python 3.11.

---

## 8. Missing Information & Status

No blocking model information is missing. All 4 model artifacts have been successfully loaded, their schemas verified, and test predictions executed.
