import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import optuna
import shap
from google.cloud import bigquery
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb

# Optuna log seviyesini azalt
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Key & BigQuery Bağlantısı
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

# 2. Veri Temizleme & Ön İşleme
target_cols = ['price', 'estimated_owners_avg', 'positive_review_percentage']
id_cols = ['app_id', 'name']

df = df.dropna(subset=target_cols).reset_index(drop=True)
base_features = [c for c in df.columns if c not in target_cols + id_cols]
df[base_features] = df[base_features].fillna(0)

print(f"✅ Temiz Veri Seti: {len(df)} Oyun, {len(base_features)} Feature")

model_save_dir = BASE_DIR / "ml_models" / "saved_models"
model_save_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 3. HIPERPARAMETRE OPTİMİZASYONU & CROSS VALIDATION FONKSİYONU
# ---------------------------------------------------------
def train_and_evaluate(X, y, model_name, target_transform=None):
    print(f"\n==================================================")
    print(f"🎯 MODEL EĞİTİMİ & OPTİMİZASYONU: {model_name}")
    print(f"==================================================")

    # Hedef değişkende log1p dönüşümü gerekiyorsa uygulayalım (Price ve Owners için sapmaları düşürür)
    y_scaled = np.log1p(y) if target_transform == 'log1p' else y

    # A. Optuna ile En İyi Parametreleri Bulma
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        
        # 3-Fold Hızlı Cross Validation
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_scaled.iloc[train_idx], y_scaled.iloc[val_idx]
            
            m = lgb.LGBMRegressor(**params)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_val)
            scores.append(mean_squared_error(y_val, preds))
            
        return np.mean(scores)

    print("🔎 Optuna ile Hiperparametre Arama Başlatılıyor (15 Trial)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15, timeout=120)
    
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['verbose'] = -1
    print(f"✨ Bulunan En İyi Parametreler: {best_params}")

    # B. 5-Fold Cross Validation ile Detaylı Performans Testi
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    r2_scores = []
    
    print("\n🔄 5-Fold Cross-Validation Çalıştırılıyor...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_scaled.iloc[train_idx], y_scaled.iloc[val_idx]
        
        fold_model = lgb.LGBMRegressor(**best_params)
        fold_model.fit(X_tr, y_tr)
        
        pred_val = fold_model.predict(X_val)
        oof_preds[val_idx] = pred_val
        
        r2_fold = r2_score(y_val, pred_val)
        r2_scores.append(r2_fold)
        print(f"   Fold {fold} R2 Skoru: {r2_fold:.4f}")

    print(f"📊 Ortalama 5-Fold R2 Skoru: {np.mean(r2_scores):.4f}")

    # C. Tüm Veriyle Final Model Eğitimi
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X, y_scaled)

    # D. Feature Importance (Özellik Katkı Analizi)
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🌟 En Çok Katkı Sağlayan İlk 10 Feature:")
    print(importance_df.head(10).to_string(index=False))

    # E. SHAP Explainer Oluşturma ve Kaydetme
    print("\n🧠 SHAP Explainer Hesaplanıyor...")
    explainer = shap.TreeExplainer(final_model)
    
    # Model, Explainer ve Parametreleri Kaydet
    joblib.dump(final_model, model_save_dir / f"model_{model_name}.pkl")
    joblib.dump(explainer, model_save_dir / f"explainer_{model_name}.pkl")
    joblib.dump(best_params, model_save_dir / f"params_{model_name}.pkl")

    return final_model, explainer, importance_df

# ---------------------------------------------------------
# MODEL EĞİTİM DÖNGÜLERİ
# ---------------------------------------------------------

# 1. PRICE MODEL (Aşırı uç değerler için log1p ölçekleme ile)
X1 = df[base_features]
y1 = df['price']
m_price, exp_price, imp_price = train_and_evaluate(X1, y1, "price", target_transform='log1p')

# 2. OWNERS MODEL (Log1p ölçekleme + Price feature ekli)
X2 = df[base_features + ['price']]
y2 = df['estimated_owners_avg']
m_owners, exp_owners, imp_owners = train_and_evaluate(X2, y2, "owners", target_transform='log1p')

# 3. REVIEW MODEL (%0-100 arası skor olduğu için direkt ölçekleme olmadan)
X3 = df[base_features + ['price']]
y3 = df['positive_review_percentage']
m_review, exp_review, imp_review = train_and_evaluate(X3, y3, "review")

print("\n🎉 TEBRİKLER! Tüm optimizasyonlar, Cross-Validation testleri, Feature Importance ve SHAP analizleri tamamlandı!")