# ChainSight-AI: Model Artifact Mapping & Source-of-Truth Documentation

**Date:** November 19, 2026  
**Auditor:** Senior MLOps Engineer & Platform Architect  
**Project:** ChainSight-AI  

---

## 1. Directory Structure Overview

The repository contains two parallel directories housing model artifacts:

1. **`models/`**:
   Dedicated directory containing clean, dedicated model artifacts intended for runtime inference, container mounting, and production deployment.
2. **`chainsight_ai_complete_models/`**:
   Comprehensive archive containing the complete training ecosystem: model artifacts, feature schema definitions (`feature_columns.json`), model evaluation metrics (`metrics.json`), training metadata (`training_metadata.json`), model comparison benchmarks (`all_models_comparison.csv`, `model_comparison.csv`), classification reports (`classification_report.txt`), and feature importance plots (`feature_importance.png`).

---

## 2. Cryptographic Hash & Byte-Level Comparison

To verify whether the artifacts in `models/` and `chainsight_ai_complete_models/` are identical or divergent iterations, a full SHA-256 cryptographic checksum and byte-size audit was executed:

| Model Task | Path in `models/` | Path in `chainsight_ai_complete_models/` | Size (Bytes) | SHA-256 Checksum | Identity Status |
|---|---|---|---|---|---|
| **Demand Forecasting** | `models/demand_forecasting/demand_model.joblib` | `chainsight_ai_complete_models/demand_forecasting/demand_model.joblib` | 262,159 | `0df47f7d32aa5b182f30b0ab91ec52754156051decdf3cdaa13f1896583e6e27` | **100% Identical** |
| **Stockout Prediction** | `models/stockout_prediction/stockout_model.joblib` | `chainsight_ai_complete_models/stockout_prediction/stockout_model.joblib` | 354,230 | `d9796fa31bbfbff61c7c148d5a0bdcab876aebc7e29588a6a33d2f27ebb9547e` | **100% Identical** |
| **Delivery ETA** | `models/delivery_eta/delivery_eta_model.joblib` | `chainsight_ai_complete_models/delivery_eta/delivery_eta_model.joblib` | 191,673 | `7133d55434e532b732b516a26550a79aa6e0761d1a6e6cdc8fc8309316983e3f` | **100% Identical** |
| **Seller Risk** | `models/seller_risk/seller_risk_model.joblib` | `chainsight_ai_complete_models/seller_risk/seller_risk_model.joblib` | 168,315 | `0ff153b3bc3126f779d55d80dbe1bcb26f1274ddb0daed6017b587738b4a1af8` | **100% Identical** |

### Key Conclusion
Every `.joblib` binary file under `models/` is bit-for-bit identical to its counterpart under `chainsight_ai_complete_models/`. There are no version discrepancies, conflicting serializations, or divergent weights between the two directories.

---

## 3. Associated Metadata & Documentation Mapping

While the model weights are identical, `chainsight_ai_complete_models/` provides crucial auxiliary metadata required for production inference validation, feature ordering, and MLflow registration:

```
ChainSight-AI/
├── models/                                      <-- PRODUCTION INFERENCE ARTIFACTS
│   ├── delivery_eta/delivery_eta_model.joblib
│   ├── demand_forecasting/demand_model.joblib
│   ├── seller_risk/seller_risk_model.joblib
│   └── stockout_prediction/stockout_model.joblib
│
└── chainsight_ai_complete_models/               <-- HISTORICAL TRAINING & REGISTRATION METADATA
    ├── delivery_eta/
    │   ├── delivery_eta_model.joblib            (Bit-identical backup)
    │   ├── feature_columns.json                 (7 input feature names)
    │   ├── metrics.json                         (MAE: 6.1624, RMSE: 7.2760, R2: -0.7749)
    │   └── training_metadata.json               (CatBoostRegressor v1.0.0, target: actual_delivery_duration)
    │
    ├── demand_forecasting/
    │   ├── demand_model.joblib                  (Bit-identical backup)
    │   ├── feature_columns.json                 (12 input feature names)
    │   ├── metrics.json                         (MAE: 0.9324, RMSE: 1.7573, WAPE: 0.7535)
    │   └── training_metadata.json               (LightGBMRegressor, hyperparameters, date ranges)
    │
    ├── seller_risk/
    │   ├── seller_risk_model.joblib             (Bit-identical backup)
    │   ├── feature_columns.json                 (7 input feature names)
    │   ├── metrics.json                         (Precision: 0.0445, Recall: 0.2236, F1: 0.0743)
    │   └── training_metadata.json               (CatBoostClassifier v1.0.0, target: delayed)
    │
    ├── stockout_prediction/
    │   ├── stockout_model.joblib                (Bit-identical backup)
    │   ├── classification_report.txt            (Precision/Recall/F1 breakdown)
    │   ├── feature_columns.json                 (13 features, types, numerical vs categorical)
    │   ├── feature_importance.png               (Plot of feature splits)
    │   ├── metrics.json                         (Accuracy: 0.9657, F1: 0.9405, ROC-AUC: 0.9901)
    │   ├── model_comparison.csv                 (Benchmark against Dummy, RF, XGBoost)
    │   └── training_metadata.json               (LightGBMClassifier v1.0.0, hyperparameters, dates)
    │
    ├── all_models_comparison.csv                (Cross-task benchmark summary)
    └── model_comparison.csv                     (Benchmark summary)
```

---

## 4. Production Inference Policy & Source of Truth

1. **Production Inference Source:**
   The application code (`src/services/` and `src/api/`) will load model artifacts directly from `models/<task>/<model_name>.joblib` (or via MLflow Model Registry `models:/<ModelName>/champion` once registered in Phase 4).
2. **Metadata & Schema Source:**
   The Pydantic input schemas, feature validation routines, and MLflow registration scripts will reference the canonical feature lists and metrics from `chainsight_ai_complete_models/`.
3. **Artifact Preservation:**
   In accordance with core project guidelines, **neither copy will be deleted or modified**. `chainsight_ai_complete_models/` remains the permanent training archive, while `models/` serves as the lightweight inference directory copied into Docker containers.
4. **Git & Docker Strategy:**
   - Model files are approximately 168 KB to 354 KB each (total < 1 MB across all 4 models). They fit easily within standard Git repository limits without requiring Git LFS.
   - Large raw datasets under `public_datasets/` (totalling > 400 MB) will be ignored by Git and Docker via `.gitignore` and `.dockerignore`.
