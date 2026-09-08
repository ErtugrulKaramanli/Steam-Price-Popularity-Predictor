import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import optuna
import shap
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.preprocessing import LabelEncoder

# Optuna loglarını sadeleştir
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ----------------------------------------------------
# XGBoost Sabit Parametreleri
# Optuna ile aranmayan, her modelde aynı kalan ayarlar.
# 'hist' LightGBM'deki histogram tabanlı bölmenin XGBoost karşılığıdır.
# ----------------------------------------------------
XGB_FIXED = {
    'random_state': 42,
    'verbosity': 0,
    'n_jobs': -1,
    'tree_method': 'hist',
    'importance_type': 'gain',
}
XGB_FIXED_REG = {**XGB_FIXED, 'objective': 'reg:squarederror'}
XGB_FIXED_CLF = {**XGB_FIXED, 'objective': 'multi:softprob', 'eval_metric': 'mlogloss'}

# ----------------------------------------------------
# Dinamik Dizin ve Veri Yolu Tespiti
# ----------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent

possible_paths = [
    BASE_DIR / "data" / "processed_steam_data.csv",
    CURRENT_DIR.parent / "data" / "processed_steam_data.csv",
    Path.cwd() / "data" / "processed_steam_data.csv"
]

data_path = None
for p in possible_paths:
    if p.exists():
        data_path = p
        break

if not data_path:
    raise FileNotFoundError("❌ Yerel veri dosyası bulunamadı!")

print(f"📂 Yerel veri yükleniyor: {data_path}")
df = pd.read_csv(data_path)

# ----------------------------------------------------
# Time Decay Sample Weights
# ----------------------------------------------------
if 'release_year' in df.columns:
    current_year = pd.Timestamp.now().year
    years_diff = (current_year - df['release_year']).fillna(5).clip(lower=0)
    weights = pd.Series(np.exp(-0.05 * years_diff), index=df.index)
    print("⏳ Time decay ağırlıkları 'release_year' üzerinden hesaplandı.")
else:
    weights = pd.Series(1.0, index=df.index)

target_cols = ['price', 'estimated_owners_avg', 'positive_review_percentage']
id_cols = [c for c in ['app_id', 'name'] if c in df.columns]
base_features = [c for c in df.columns if c not in target_cols + id_cols]

model_save_dir = BASE_DIR / "ml_models" / "saved_models_xgb"
model_save_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------
# 1. REGRESYON EĞİTİM FONKSİYONU (Price ve Review İçin)
# ----------------------------------------------------
def train_and_save_regressor(X, y, model_name, target_transform=None, sample_weights=None):
    print(f"\n==================================================")
    print(f"🎯 REGRESYON MODELİ EĞİTİLİYOR: {model_name.upper()}")
    print(f"==================================================")

    y_scaled = np.log1p(y) if target_transform == 'log1p' else y

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y_scaled, sample_weights, test_size=0.2, random_state=42
    )

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            **XGB_FIXED_REG
        }
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_tr = w_train.iloc[train_idx]
            
            m = xgb.XGBRegressor(**params)
            m.fit(X_tr, y_tr, sample_weight=w_tr)
            preds = m.predict(X_val)
            scores.append(mean_squared_error(y_val, preds))
            
        return np.mean(scores)

    print("🔎 Optuna Hiperparametre Araması Başlatılıyor...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15, timeout=120)
    
    best_params = study.best_params
    best_params.update(XGB_FIXED_REG)

    initial_model = xgb.XGBRegressor(**best_params)
    initial_model.fit(X_train, y_train, sample_weight=w_train)

    importance = pd.Series(initial_model.feature_importances_, index=X.columns)
    selected_features = importance[importance > 0].index.tolist()
    
    print(f"✂️ Feature Selection: Toplam {len(X.columns)} değişkenden {len(selected_features)} önemli değişken seçildi.")

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train[selected_features], y_train, sample_weight=w_train)

    explainer = shap.TreeExplainer(final_model)

    joblib.dump(final_model, model_save_dir / f"model_{model_name}.pkl")
    joblib.dump(explainer, model_save_dir / f"explainer_{model_name}.pkl")
    joblib.dump((X_test[selected_features], y_test), model_save_dir / f"test_data_{model_name}.pkl")
    
    print(f"✅ {model_name.upper()} modeli eğitildi ve kaydedildi!")

# ----------------------------------------------------
# 2. SINIFLANDIRMA EĞİTİM FONKSİYONU (Owners İçin)
# ----------------------------------------------------
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def train_and_save_classifier(X, y_raw, model_name, sample_weights=None):
    print(f"\n==================================================")
    print(f"🎯 SINIFLANDIRMA MODELİ EĞİTİLİYOR: {model_name.upper()}")
    print(f"==================================================")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw.astype(str))
    
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y_encoded, sample_weights, test_size=0.2, random_state=42, stratify=y_encoded
    )

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            **XGB_FIXED_CLF
        }
        
        # Sınıf dağılımlarını korumak için StratifiedKFold kullanıyoruz
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            w_tr = w_train.iloc[train_idx]
            
            m = xgb.XGBClassifier(**params)
            m.fit(X_tr, y_tr, sample_weight=w_tr)
            preds = m.predict(X_val)
            # log_loss yerine katman bazlı accuracy_score kullanıyoruz
            scores.append(accuracy_score(y_val, preds))
            
        return np.mean(scores)

    print("🔎 Optuna Hiperparametre Araması Başlatılıyor...")
    # Accuracy maximize edileceği için direction="maximize"
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15, timeout=120)
    
    best_params = study.best_params
    best_params.update(XGB_FIXED_CLF)

    initial_model = xgb.XGBClassifier(**best_params)
    initial_model.fit(X_train, y_train, sample_weight=w_train)

    importance = pd.Series(initial_model.feature_importances_, index=X.columns)
    selected_features = importance[importance > 0].index.tolist()
    
    print(f"✂️ Feature Selection: Toplam {len(X.columns)} değişkenden {len(selected_features)} önemli değişken seçildi.")

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train[selected_features], y_train, sample_weight=w_train)

    explainer = shap.TreeExplainer(final_model)

    joblib.dump(final_model, model_save_dir / f"model_{model_name}.pkl")
    joblib.dump(explainer, model_save_dir / f"explainer_{model_name}.pkl")
    joblib.dump((X_test[selected_features], y_test), model_save_dir / f"test_data_{model_name}.pkl")
    joblib.dump(label_encoder, model_save_dir / f"label_encoder_{model_name}.pkl")
    
    print(f"✅ {model_name.upper()} Sınıflandırma modeli eğitildi ve kaydedildi!")
# ----------------------------------------------------
# MODEL EĞİTİMLERİ
# ----------------------------------------------------

# 1. PRICE MODELİ (Regresyon)
X1 = df[base_features]
train_and_save_regressor(X1, df['price'], "price", target_transform='log1p', sample_weights=weights)

# 2. OWNERS MODELİ (Sınıflandırma)
X2 = df[base_features + ['price']]
train_and_save_classifier(X2, df['estimated_owners_avg'], "owners", sample_weights=weights)

# 3. REVIEW MODELİ (Regresyon)
X3 = df[base_features + ['price']]
train_and_save_regressor(X3, df['positive_review_percentage'], "review", sample_weights=weights)

print("\n🎉 Tüm optimize edilmiş modeller başarıyla eğitildi!")