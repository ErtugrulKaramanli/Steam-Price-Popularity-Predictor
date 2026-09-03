import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import optuna
import shap
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "processed_steam_data.parquet"

if not data_path.exists():
    raise FileNotFoundError("❌ Yerel veri dosyası bulunamadı! Önce 'python ml_models/1_get_data.py' çalıştırın.")

print("📂 Yerel Parquet verisi yükleniyor...")
df = pd.read_parquet(data_path)

target_cols = ['price', 'estimated_owners_avg', 'positive_review_percentage']
id_cols = ['app_id', 'name']
base_features = [c for c in df.columns if c not in target_cols + id_cols]

model_save_dir = BASE_DIR / "ml_models" / "saved_models"
model_save_dir.mkdir(parents=True, exist_ok=True)

def train_and_save(X, y, model_name, target_transform=None):
    print(f"\n==================================================")
    print(f"🎯 MODEL EĞİTİLİYOR & OPTİMİZE EDİLİYOR: {model_name.upper()}")
    print(f"==================================================")

    y_scaled = np.log1p(y) if target_transform == 'log1p' else y

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

    print("🔎 Optuna Hiperparametre Araması Yapılıyor...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15, timeout=120)
    
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['verbose'] = -1

    # Train / Test Ayrımı
    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train)

    print("🧠 SHAP Explainer Hesaplanıyor...")
    explainer = shap.TreeExplainer(final_model)

    # Kayıtlar
    joblib.dump(final_model, model_save_dir / f"model_{model_name}.pkl")
    joblib.dump(explainer, model_save_dir / f"explainer_{model_name}.pkl")
    joblib.dump((X_test, y_test), model_save_dir / f"test_data_{model_name}.pkl")
    
    print(f"✅ {model_name.upper()} modeli ve test verisi kaydedildi!")

# Model Eğitimleri
X1 = df[base_features]
train_and_save(X1, df['price'], "price", target_transform='log1p')

X2 = df[base_features + ['price']]
train_and_save(X2, df['estimated_owners_avg'], "owners", target_transform='log1p')

X3 = df[base_features + ['price']]
train_and_save(X3, df['positive_review_percentage'], "review")

print("\n🎉 Tüm modeller eğitildi ve yerel olarak kaydedildi!")