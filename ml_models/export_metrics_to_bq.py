from pathlib import Path
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

# -------------------------------------------------------------------------
# 1. ORTAM VE DİZİN YAPILANMASI
# -------------------------------------------------------------------------
base_dir = Path(__file__).resolve().parent

# -------------------------------------------------------------------------
# 2. BIGQUERY BAĞLANTISI VE CSV OKUMA
# -------------------------------------------------------------------------
print("🚀 BigQuery bağlantısı kuruluyor ve metrik raporu okunuyor...")
key_path = base_dir.parent / "bigquery_key.json"
credentials = service_account.Credentials.from_service_account_file(key_path)
client = bigquery.Client(
    credentials=credentials, project="datascientiststeamproject"
)

csv_path = base_dir / "ml_model_metrics_report_v2.csv"

if not csv_path.exists():
    raise FileNotFoundError(f"Metrik raporu bulunamadı: {csv_path}. Önce ana modeli çalıştırıp CSV'yi oluşturmalısın.")

df_metrics = pd.read_csv(csv_path)
print(f"✅ Metrik raporu başarıyla okundu ({len(df_metrics)} satır).")

# -------------------------------------------------------------------------
# 3. BİGQUERY'E TABLO OLARAK AKTARMA
# -------------------------------------------------------------------------
dataset_id = "dbt_dataset_steam"
table_name = "viz_model_metrics_v2"
table_ref = f"{client.project}.{dataset_id}.{table_name}"

job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")

job = client.load_table_from_dataframe(df_metrics, table_ref, job_config=job_config)
job.result()

print(f"🎉 `{table_name}` tablosu BigQuery'e başarıyla yüklendi!")