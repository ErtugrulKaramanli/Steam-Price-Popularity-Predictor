import os
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

warnings.filterwarnings("ignore")

# 1. Dizin ve Bağlantı Ayarları
base_dir = Path(__file__).resolve().parent
model_save_dir = base_dir / "saved_models"
key_path = base_dir.parent / "bigquery_key.json"

credentials = service_account.Credentials.from_service_account_file(key_path)
client = bigquery.Client(
    credentials=credentials, project="datascientiststeamproject"
)
dataset_id = "dbt_dataset_steam"

# -------------------------------------------------------------------------
# A. REGRESYON GRAFİKLERİ İÇİN VERİ (Price & Review)
# -> Looker Studio: Scatter Plot (Actual vs Predicted) & Line/Bar (Residuals)
# -------------------------------------------------------------------------
print("📊 Regresyon modellerinin tahmin verileri hazırlanıyor...")
reg_records = []

# Price Modeli
if (model_save_dir / "bundle_price.pkl").exists():
    bundle_p = joblib.load(model_save_dir / "bundle_price.pkl")
    X_test_p, y_test_p = joblib.load(model_save_dir / "test_data_price.pkl")
    preds_log_p = bundle_p["model"].predict(X_test_p)
    preds_p = np.expm1(preds_log_p)

    for actual, pred in zip(y_test_p, preds_p):
        reg_records.append(
            {
                "model_name": "Price Model",
                "actual_value": float(actual),
                "predicted_value": float(pred),
                "residual": float(actual - pred),
                "abs_error": float(abs(actual - pred)),
            }
        )

# Review Modeli
if (model_save_dir / "bundle_review.pkl").exists():
    bundle_r = joblib.load(model_save_dir / "bundle_review.pkl")
    X_test_r, y_test_r = joblib.load(model_save_dir / "test_data_review.pkl")
    preds_r = bundle_r["model"].predict(X_test_r)

    for actual, pred in zip(y_test_r, preds_r):
        reg_records.append(
            {
                "model_name": "Review Model",
                "actual_value": float(actual),
                "predicted_value": float(pred),
                "residual": float(actual - pred),
                "abs_error": float(abs(actual - pred)),
            }
        )

df_reg_viz = pd.DataFrame(reg_records)

# -------------------------------------------------------------------------
# B. SINIFLANDIRMA GRAFİKLERİ İÇİN VERİ (Owners Model - Confusion Matrix)
# -> Looker Studio: Heatmap / Pivot Table
# -------------------------------------------------------------------------
print("👥 Sınıflandırma modelinin tahmin verileri hazırlanıyor...")
clf_records = []

if (model_save_dir / "bundle_owners.pkl").exists():
    bundle_o = joblib.load(model_save_dir / "bundle_owners.pkl")
    X_test_o, y_test_o = joblib.load(model_save_dir / "test_data_owners.pkl")
    preds_o = bundle_o["model"].predict(X_test_o)

    for actual, pred in zip(y_test_o, preds_o):
        clf_records.append(
            {
                "model_name": "Owners Model",
                "actual_class": str(actual),
                "predicted_class": str(pred),
                "is_correct": int(actual == pred),
            }
        )

df_clf_viz = pd.DataFrame(clf_records)

# -------------------------------------------------------------------------
# C. ÖZNİTELİK ÖNEMİ (Feature Importance) VERİSİ
# -> Looker Studio: Bar Chart (Top Features)
# -------------------------------------------------------------------------
print("🔑 Değişken önem skorları (Feature Importance) çekiliyor...")
feat_records = []

for model_name in ["price", "owners", "review"]:
    pkl_path = model_save_dir / f"bundle_{model_name}.pkl"
    if pkl_path.exists():
        bundle = joblib.load(pkl_path)
        model = bundle["model"]
        features = bundle["selected_features"]
        importances = model.feature_importances_

        for feat, imp in zip(features, importances):
            feat_records.append(
                {
                    "model_name": f"{model_name.capitalize()} Model",
                    "feature_name": feat,
                    "importance_score": float(imp),
                }
            )

df_feat_viz = pd.DataFrame(feat_records)

# -------------------------------------------------------------------------
# BIGQUERY'E YÜKLEME SÜRECİ
# -------------------------------------------------------------------------
print("\n🚀 BigQuery tabloları güncelleniyor...")

tables = {
    "viz_regression_predictions": df_reg_viz,
    "viz_classification_predictions": df_clf_viz,
    "viz_feature_importances": df_feat_viz,
}

for table_name, df_data in tables.items():
    table_ref = f"{client.project}.{dataset_id}.{table_name}"
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE"
    )  # Her çalıştırmada günceller

    job = client.load_table_from_dataframe(
        df_data, table_ref, job_config=job_config
    )
    job.result()  # Yüklemenin bitmesini bekle
    print(
        f"✅ {table_name} tablosu yüklendi! (Toplam Satır: {len(df_data)})"
    )

print(
    "\n🎉 Tüm veriler BigQuery'ye aktarıldı. Looker Studio'ya bağlanmaya hazır!"
)