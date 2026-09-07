import pandas as pd
import numpy as np
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent

def find_data_file():
    all_files = []
    for ext in ['*.csv', '*.parquet']:
        for f in base_dir.rglob(ext):
            if 'venv' not in f.parts and '.env' not in f.parts and '.git' not in f.parts:
                all_files.append(f)
    for f in all_files:
        if any(k in f.name.lower() for k in ['clean', 'prep', 'steam', 'game']):
            return f
    return all_files[0] if all_files else None

def analyze_updated_distributions():
    target_file = find_data_file()
    if not target_file:
        print("❌ Veri dosyası bulunamadı!")
        return

    print(f"✅ Okunan Dosya: {target_file.relative_to(base_dir)}")
    df = pd.read_parquet(target_file) if target_file.suffix == ".parquet" else pd.read_csv(target_file)

    owners_col = "estimated_owners_avg"
    price_col = "price"

    # ----------------------------------------------------
    # 1. YENİ BİRLEŞTİRİLMİŞ OWNERS DAĞILIMI (Paid Oyunlar)
    # ----------------------------------------------------
    print("\n==================================================")
    print("📊 ÖNERİLEN 5 SINIFLI OWNERS DAĞILIMI (Price > 0)")
    print("==================================================")
    
    df_paid = df[df[price_col] > 0].copy()

    # Sınıf eşleme mantığı
    def map_owners_group(val):
        if val <= 10000:
            return "1. 0 - 10k (Low)"
        elif val <= 75000:
            return "2. 10k - 75k (Moderate)"
        elif val <= 350000:
            return "3. 75k - 350k (Hit)"
        elif val <= 1500000:
            return "4. 350k - 1.5M (Breakthrough)"
        else:
            return "5. 1.5M+ (Global Hit)"

    df_paid['owners_grouped'] = df_paid[owners_col].apply(map_owners_group)
    
    grouped_counts = df_paid['owners_grouped'].value_counts().sort_index()
    grouped_percents = (df_paid['owners_grouped'].value_counts(normalize=True).sort_index()) * 100

    owners_summary = pd.DataFrame({
        'Oyun Sayısı': grouped_counts,
        'Yüzde (%)': grouped_percents.round(2),
        'Kümülatif Yüzde (%)': grouped_percents.cumsum().round(2)
    })
    print(owners_summary.to_string())

    # ----------------------------------------------------
    # 2. ÖZELLEŞTİRİLMİŞ PRICE DAĞILIMI (0$ Ayrılmış)
    # ----------------------------------------------------
    print("\n==================================================")
    print("💰 ÖZELLEŞTİRİLMİŞ FIYAT ARALIKLARI DAĞILIMI")
    print("==================================================")

    # Fiyat Bins: 0 (F2P), 0.01-5$, 5-10$, 10-15$, 15-20$, 20-30$, 30-50$, 50$+
    bins = [-0.01, 0.001, 5, 10, 15, 20, 30, 50, df[price_col].max()]
    labels = [
        "0$ (Free to Play)",
        "0.01$ - 5.00$",
        "5.01$ - 10.00$",
        "10.01$ - 15.00$",
        "15.01$ - 20.00$",
        "20.01$ - 30.00$",
        "30.01$ - 50.00$",
        "50.00$+"
    ]

    df['price_custom_bin'] = pd.cut(df[price_col], bins=bins, labels=labels)
    
    price_counts = df['price_custom_bin'].value_counts().sort_index()
    price_percents = (df['price_custom_bin'].value_counts(normalize=True).sort_index()) * 100

    price_summary = pd.DataFrame({
        'Oyun Sayısı': price_counts,
        'Yüzde (%)': price_percents.round(2),
        'Kümülatif Yüzde (%)': price_percents.cumsum().round(2)
    })
    print(price_summary.to_string())

if __name__ == "__main__":
    analyze_updated_distributions()