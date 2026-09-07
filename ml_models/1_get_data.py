import os
import pandas as pd
import numpy as np
from pathlib import Path
from google.cloud import bigquery
from google.oauth2 import service_account

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# BigQuery Bağlantı Bilgileri
PROJECT_ID = "datascientiststeamproject"
DATASET_ID = "dbt_dataset_steam"
TABLE_ID = "ml_features_steam"
KEY_PATH = base_dir / "bigquery_key.json"

def map_owners_group(val):
    if pd.isna(val):
        return np.nan
    val = float(val)
    if val <= 10000:
        return "0-10k"
    elif val <= 75000:
        return "10k-75k"
    elif val <= 350000:
        return "75k-350k"
    elif val <= 1500000:
        return "350k-1.5M"
    else:
        return "1.5M+"

def fetch_and_prepare_data():
    print("🚀 BigQuery'ye bağlanılıyor...")
    if not KEY_PATH.exists():
        raise FileNotFoundError(f"❌ Key dosyası bulunamadı: {KEY_PATH}")

    credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

    query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    """
    
    print("📥 Veri BigQuery'den çekiliyor...")
    df = client.query(query).to_dataframe()
    print(f"🔹 Çekilen Toplam Oyun Sayısı: {len(df):,}")

    # 1. F2P ve Ücretsiz Oyunları Temizleme
    df = df[df['price'] > 0].copy()
    if 'genre_free_to_play' in df.columns:
        df = df[df['genre_free_to_play'] == 0].copy()
    print(f"✂️ Ücretsiz (F2P) oyunlar elendi. Kalan: {len(df):,}")

    # 2. Owners Sınıflarını 5 Segmente Dönüştürme
    df['owners_grouped'] = df['estimated_owners_avg'].apply(map_owners_group)

    # 3. Eksik Veri Temizliği
    df = df.dropna(subset=['price', 'owners_grouped', 'positive_review_percentage']).copy()

    # 4. Parquet Olarak Kaydetme
    output_path = data_dir / "processed_steam_data.parquet"
    df.to_parquet(output_path, index=False)
    print(f"💾 İşlenmiş veri başarıyla kaydedildi: {output_path.relative_to(base_dir)}")

if __name__ == "__main__":
    fetch_and_prepare_data()