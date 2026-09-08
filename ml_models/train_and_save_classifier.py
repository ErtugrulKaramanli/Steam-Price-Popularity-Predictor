import os
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent

# ----------------------------------------------------
# 1. BIGQUERY VERİ ÇEKME ADIMI
# ----------------------------------------------------
PROJECT_ID = "datascientiststeamproject"
DATASET_NAME = "dbt_dataset_steam"
TABLE_NAME = "ml_features_steam"

KEY_PATH = BASE_DIR / "bigquery_key.json"

if not KEY_PATH.exists():
    raise FileNotFoundError(f"❌ BigQuery anahtar dosyası bulunamadı: {KEY_PATH}")

print("📂 BigQuery'den güncel dbt mart verisi çekiliyor...")

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_NAME}.{TABLE_NAME}`"
df = client.query(query).to_dataframe()

print(f"✅ Veri başarıyla çekildi! Satır Sayısı: {df.shape[0]}, Sütun Sayısı: {df.shape[1]}")

# ----------------------------------------------------
# 2. ÖN İŞLEME VE DEĞİŞKEN TANIMLARI
# ----------------------------------------------------

# Time Decay Sample Weights
if "release_year" in df.columns:
    current_year = pd.Timestamp.now().year
    years_diff = (current_year - df["release_year"]).fillna(5).clip(lower=0)
    weights = pd.Series(np.exp(-0.05 * years_diff), index=df.index)
    print("⏳ Time decay ağırlıkları 'release_year' üzerinden hesaplandı.")
else:
    weights = pd.Series(1.0, index=df.index)

# tüm Hedef Değişkenler
target_cols = ["price", "target_owners_avg", "target_owner_class", "positive_review_percentage"]

# ID ve Metin Kolonları
id_cols = [c for c in ["app_id", "name", "developer", "publisher"] if c in df.columns]

# Temel Feature Listesi
base_features = [c for c in df.columns if c not in target_cols + id_cols]

# String/Object tipindeki kategorik sütunları LightGBM için 'category' tipine çevir
for col in base_features:
    if df[col].dtype == "object":
        df[col] = df[col].astype("category")

model_save_dir = BASE_DIR / "ml_models" / "saved_models"
model_save_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------
# 3. MODEL EĞİTİM FONKSİYONLARI
# ----------------------------------------------------

def train_and_save_regressor(X, y, model_name, target_transform=None, sample_weights=None):
    print(f"\n==================================================")
    print(f"🎯 REGRESYON MODELİ EĞİTİLİYOR: {model_name.upper()}")
    print(f"==================================================")

    y_scaled = np.log1p(y) if target_transform == "log1p" else y

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y_scaled, sample_weights, test_size=0.2, random_state=42
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "verbose": -1,
        }

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_tr = w_train.iloc[train_idx]

            m = lgb.LGBMRegressor(**params)
            m.fit(X_tr, y_tr, sample_weight=w_tr)
            preds = m.predict(X_val)
            scores.append(mean_squared_error(y_val, preds))

        return np.mean(scores)

    print("🔎 Optuna Hiperparametre Araması Başlatılıyor...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15, timeout=120)

    best_params = study.best_params
    best_params["random_state"] = 42
    best_params["verbose"] = -1

    initial_model = lgb.LGBMRegressor(**best_params)
    initial_model.fit(X_train, y_train, sample_weight=w_train)

    importance = pd.Series(initial_model.feature_importances_, index=X.columns)
    selected_features = importance[importance > 0].index.tolist()

    print(f"✂️ Feature Selection: Toplam {len(X.columns)} değişkenden {len(selected_features)} önemli değişken seçildi.")

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train[selected_features], y_train, sample_weight=w_train)

    joblib.dump(final_model, model_save_dir / f"model_{model_name}.pkl")
    joblib.dump((X_test[selected_features], y_test), model_save_dir / f"test_data_{model_name}.pkl")

    print(f"✅ {model_name.upper()} modeli eğitildi ve kaydedildi!")


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
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "objective": "multiclass",
            "random_state": 42,
            "verbose": -1,
        }

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            w_tr = w_train.iloc[train_idx]

            m = lgb.LGBMClassifier(**params)
            m.fit(X_tr, y_tr, sample_weight=w_tr)
            preds = m.predict(X_val)
            scores.append(f1_score(y_val, preds, average="macro"))

        return np.mean(scores)

    print("🔎 Optuna Hiperparametre Araması Başlatılıyor...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15, timeout=120)

    best_params = study.best_params
    best_params["objective"] = "multiclass"
    best_params["random_state"] = 42
    best_params["verbose"] = -1

    initial_model = lgb.LGBMClassifier(**best_params)
    initial_model.fit(X_train, y_train, sample_weight=w_train)

    importance = pd.Series(initial_model.feature_importances_, index=X.columns)
    selected_features = importance[importance > 0].index.tolist()

    print(f"✂️ Feature Selection: Toplam {len(X.columns)} değişkenden {len(selected_features)} önemli değişken seçildi.")

    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train[selected_features], y_train, sample_weight=w_train)

    joblib.dump(final_model, model_save_dir / f"model_{model_name}.pkl")
    joblib.dump((X_test[selected_features], y_test), model_save_dir / f"test_data_{model_name}.pkl")
    joblib.dump(label_encoder, model_save_dir / f"label_encoder_{model_name}.pkl")

    print(f"✅ {model_name.upper()} Sınıflandırma modeli eğitildi ve kaydedildi!")


# ----------------------------------------------------
# 4. MODEL EĞİTİMLERİ
# ----------------------------------------------------

# 1. PRICE MODELİ (7'li Sınıflandırma)
bins = [0.48, 2.99, 6.99, 11.99, 16.99, 24.99, 39.99, 100.0]
labels = [
    "Micro_Budget",   # $0.49 - $2.99
    "Budget",         # $3.00 - $6.99
    "Indie_Starter",  # $7.00 - $11.99
    "Indie_Mid",      # $12.00 - $16.99
    "Indie_Premium",  # $17.00 - $24.99
    "AA_Mid",         # $25.00 - $39.99
    "AAA_Flagship"    # $40.00 - $100.00
]

df_price = df[(df["price"] >= 0.49) & (df["price"] <= 100.0)].copy()
df_price["target_price_class"] = pd.cut(df_price["price"], bins=bins, labels=labels)

X1 = df_price[base_features]
w1 = weights.loc[df_price.index]

print(f"\n✂️ Price Modeli 7 gruba bölündü. Satır sayısı: {len(df_price)}")
train_and_save_classifier(X1, df_price["target_price_class"], "price", sample_weights=w1)

# 2. OWNERS MODELİ
X2 = df[base_features + ["price"]]
train_and_save_classifier(X2, df["target_owner_class"], "owners", sample_weights=weights)

# 3. REVIEW MODELİ
X3 = df[base_features + ["price"]]
train_and_save_regressor(X3, df["positive_review_percentage"], "review", sample_weights=weights)

print("\n🎉 Tüm optimize edilmiş modeller başarıyla eğitildi!")

# ----------------------------------------------------
# 5. OTOMATİK MODEL DEĞERLENDİRME ADIMI
# ----------------------------------------------------

def load_artifacts(model_name):
    model = joblib.load(model_save_dir / f"model_{model_name}.pkl")
    X_test, y_test = joblib.load(model_save_dir / f"test_data_{model_name}.pkl")
    return model, X_test, y_test


def evaluate_all_models():
    print("\n📊 MODEL DEĞERLENDİRME SONUÇLARI\n" + "-" * 40)
    results = []

    # 1. Price (Sınıflandırma)
    model, X_test, y_test_enc = load_artifacts("price")
    preds_enc = model.predict(X_test)

    acc_p = accuracy_score(y_test_enc, preds_enc)
    f1_p = f1_score(y_test_enc, preds_enc, average="macro")
    print(f"💰 PRICE (Class) -> Accuracy: {acc_p:.4f} | Macro F1: {f1_p:.4f}")

    results.append({
        "model": "price_class", "r2": np.nan, "mae": np.nan, "rmse": np.nan, "accuracy": acc_p, "macro_f1": f1_p
    })

    # 2. Owners (Sınıflandırma)
    model, X_test, y_test_enc = load_artifacts("owners")
    preds_enc = model.predict(X_test)

    acc_o = accuracy_score(y_test_enc, preds_enc)
    f1_o = f1_score(y_test_enc, preds_enc, average="macro")
    print(f"👥 OWNERS        -> Accuracy: {acc_o:.4f} | Macro F1: {f1_o:.4f}")

    results.append({
        "model": "owners", "r2": np.nan, "mae": np.nan, "rmse": np.nan, "accuracy": acc_o, "macro_f1": f1_o
    })

    # 3. Review (Regresyon)
    model, X_test, y_test = load_artifacts("review")
    preds = np.clip(model.predict(X_test), 0, 100)

    r2_r = r2_score(y_test, preds)
    mae_r = mean_absolute_error(y_test, preds)
    rmse_r = np.sqrt(mean_squared_error(y_test, preds))
    print(f"⭐ REVIEW        -> R²: {r2_r:.4f} | MAE: %{mae_r:.2f} puan | RMSE: %{rmse_r:.2f} puan")

    results.append({
        "model": "review", "r2": r2_r, "mae": mae_r, "rmse": rmse_r, "accuracy": np.nan, "macro_f1": np.nan
    })

    print("\n" + "=" * 60)
    print("📌 GENEL METRİK ÖZETİ")
    print("=" * 60)

    summary_df = pd.DataFrame(results).set_index("model")
    print(summary_df.to_string())

    summary_path = model_save_dir / "evaluation_summary_7class.csv"
    summary_df.to_csv(summary_path)
    print(f"\n💾 Özet tablo kaydedildi: {summary_path}")


if __name__ == "__main__":
    evaluate_all_models()