import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path.cwd()
KEY_PATH = BASE_DIR / "bigquery_key.json"

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project="datascientiststeamproject")

query = "SELECT price FROM `datascientiststeamproject.dbt_dataset_steam.ml_features_steam`"
df = client.query(query).to_dataframe()

print("📊 TEMEL İSTATİSTİKLER")
print(df['price'].describe())

print("\n🎯 YÜZDE ARALIKLARI (PERCENTILES)")
print(df['price'].quantile([0.1, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]))

print("\nÖZEL DURUM SAYILARI")
print(f"Ücretsiz Oyunlar (0$): {(df['price'] == 0).sum()} tane (%{(df['price'] == 0).mean()*100:.1f})")
print(f"100$ Üstü Oyunlar: {(df['price'] > 100).sum()} tane")