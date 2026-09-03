import os
from google.cloud import bigquery
import pandas as pd

# 1. Google Cloud Kimlik Doğrulaması (İndirdiğin JSON dosyasının adı)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "bigquery_key.json"

# 2. BigQuery Client Başlatma
client = bigquery.Client()

# 3. SQL Sorgusu (Proje ID ve Dataset adını BigQuery'deki isimlerinle değiştir)
# Örn: `datascientiststeamproject.dbt_marts.ml_features_steam`
QUERY = """
    SELECT * 
    FROM `datascientiststeamproject.dbt_marts.ml_features_steam`
"""

print("BigQuery'den veriler çekiliyor, lütfen bekleyin...")

# 4. Veriyi DataFrame olarak yükleme
df = client.query(QUERY).to_dataframe()

print(f"\nSUCCESS! Veri başarıyla çekildi.")
print(f"Toplam Satır (Oyun Sayısı): {len(df)}")
print(f"Toplam Sütun (Feature Sayısı): {len(df.columns)}")

# Sütun isimlerini ve ilk 3 satırı kontrol edelim
print("\nÖrnek Veri:")
print(df.head(3))