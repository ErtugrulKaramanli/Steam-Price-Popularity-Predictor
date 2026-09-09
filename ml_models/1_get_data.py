import os
import pandas as pd
import numpy as np
from google.cloud import bigquery

# ---------------------------------------------------------
# 1. BigQuery Veri Çekme Aşaması
# ---------------------------------------------------------
def fetch_data_from_bigquery():
    print("BigQuery'den güncel mart verisi çekiliyor...")
    
    # GCP BigQuery Client başlatma (Ortam değişkenleriniz yapılandırılmış varsayıldı)
    client = bigquery.Client()
    
    # dbt mart modelinizin bulunduğu dataset ve tablo adı
    query = """
        SELECT * 
        FROM `your_project.your_dataset.ml_features_steam`
    """
    
    df = client.query(query).to_dataframe()
    print(f"Toplam Çekilen Ham Satır Sayısı: {len(df)}")
    return df


# ---------------------------------------------------------
# 2. Oyuncu Sayısı Sınıflandırma (Homojen Segmentasyon)
# ---------------------------------------------------------
def map_owners_to_classes(raw_range):
    """
    dbt'den gelen 'estimated_owners_raw' (örn: '0 - 20000', '20000 - 50000') 
    metinsel ifadelerini homojen makine öğrenmesi sınıflarına dönüştürür.
    """
    if pd.isna(raw_range):
        return np.nan
    
    val = str(raw_range).strip().lower()
    
    # Kaggle Steam Dataset standart range eşlemeleri
    if val in ['0 - 20000', '0 .. 20,000', '0 - 0', '0 .. 0']:
        return '0-20k'
    elif val in ['20000 - 50000', '20,000 .. 50,000']:
        return '20k-50k'
    elif val in ['50000 - 100000', '50,000 .. 100,000']:
        return '50k-100k'
    elif val in ['100000 - 200000', '100,000 .. 200,000']:
        return '100k-200k'
    elif val in ['200000 - 500000', '200,000 .. 500,000']:
        return '200k-500k'
    elif val in ['500000 - 1000000', '500,000 .. 1,000,000']:
        return '500k-1M'
    else:
        # 1M+ üstü daha nadir büyük projeleri kapsar
        return '1M+'


# ---------------------------------------------------------
# 3. Ana İşleme Fonksiyonu
# ---------------------------------------------------------
def process_data(df):
    print("Veri ön işleme ve hedef değişken (target) dönüşümleri yapılıyor...")
    
    # Null değer güvenliği (Kritik kolonlarda)
    df = df.dropna(subset=['price', 'estimated_owners_raw']).copy()
    
    # 1. Sınıflandırma Target'ını Oluşturma
    df['target_owner_class'] = df['estimated_owners_raw'].apply(map_owners_to_classes)
    
    # 2. Ham/Gereksiz Identifier veya Text Kolonlarını Ayıklama
    # app_id ve name EDA / raporlama için saklanabilir, eğitim öncesi düşürülür
    drop_cols = ['estimated_owners_raw']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Sınıf Dağılımını Ekrana Basalım (Class Imbalance Kontrolü)
    print("\nYeni Sınıf Dağılımı (target_owner_class):")
    print(df['target_owner_class'].value_counts(dropna=False))
    
    return df


# ---------------------------------------------------------
# 4. Pipeline Çalıştırma & Kaydetme
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Veriyi Çek
    raw_df = fetch_data_from_bigquery()
    
    # 2. İşle
    processed_df = process_data(raw_df)
    
    # 3. Veriyi Yerel Dizine Kaydet
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "steam_ml_ready.parquet")
    
    # Parquet formatı veri tiplerini koruduğu için CSV'ye göre tercih edilir
    processed_df.to_parquet(output_path, index=False)
    print(f"\nİşlenmiş veri başarıyla kaydedildi: {output_path}")
    print(f"Final Veri Boyutu: {processed_df.shape}")