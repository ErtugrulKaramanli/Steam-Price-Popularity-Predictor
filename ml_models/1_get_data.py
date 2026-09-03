import os
from pathlib import Path
from google.cloud import bigquery
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
KEY_PATH = BASE_DIR / "bigquery_key.json"

if not KEY_PATH.exists():
    raise FileNotFoundError(f"❌ Key dosyası bulunamadı: {KEY_PATH}")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(KEY_PATH)
client = bigquery.Client()

QUERY = """
    SELECT * 
    FROM `datascientiststeamproject.dbt_dataset_steam.ml_features_steam`
"""

print("🚀 BigQuery'den ml_features_steam tablosu çekiliyor...")
query_job = client.query(QUERY, location="EU")
df = query_job.to_dataframe()

# Veri Temizleme
target_cols = ['price', 'estimated_owners_avg', 'positive_review_percentage']
id_cols = ['app_id', 'name']

df = df.dropna(subset=target_cols).reset_index(drop=True)
base_features = [c for c in df.columns if c not in target_cols + id_cols]
df[base_features] = df[base_features].fillna(0)

# Veriyi Yerel Diskimize Parquet Formatında Kaydetme (Çok Hızlıdır)
data_dir = BASE_DIR / "data"
data_dir.mkdir(exist_ok=True)
output_path = data_dir / "processed_steam_data.parquet"

df.to_parquet(output_path, index=False)
print(f"✅ Veri yerel ortama kaydedildi: {output_path} ({len(df)} Oyun)")