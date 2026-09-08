import os
from pathlib import Path
import warnings
import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -------------------------------------------------------------------------
# 1. ORTAMA VE DİZİN YAPILANMASI
# -------------------------------------------------------------------------
base_dir = Path(__file__).resolve().parent
model_save_dir = base_dir / "saved_models"
model_save_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 2. BIGQUERY VERİ ÇEKME
# -------------------------------------------------------------------------
print("🚀 BigQuery bağlantısı kuruluyor ve veriler yükleniyor...")
key_path = base_dir.parent / "bigquery_key.json"
credentials = service_account.Credentials.from_service_account_file(key_path)
client = bigquery.Client(
    credentials=credentials, project="datascientiststeamproject"
)

query = "SELECT * FROM dbt_dataset_steam.ml_features_steam"
df = client.query(query).to_dataframe()
print(f"✅ Toplam {len(df)} satır veri BigQuery'den çekildi.")

# Target ve Id kolonlarının ayıklanması
exclude_cols = [
    "appid",
    "name",
    "release_date",
    "price",
    "positive_review_percentage",
    "target_owner_class",
]
base_features = [c for c in df.columns if c not in exclude_cols]

# Zaman Ağrılıklandırması (Time Decay)
if "release_year" in df.columns:
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(2020)
elif "release_date" in df.columns:
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(2020)
else:
    df["release_year"] = 2020

current_year = 2026
decay_rate = 0.05
weights = np.exp(-decay_rate * (current_year - df["release_year"]))


# -------------------------------------------------------------------------
# 3. REGRESYON MODEL EĞİTİM FONKSİYONU (PRICE & REVIEW)
# -------------------------------------------------------------------------
def train_and_save_regressor(
    X, y, model_name, target_transform=None, sample_weights=None
):
    print(f"\n==================================================")
    print(f"🎯 REGRESYON MODELİ EĞİTİLİYOR: {model_name.upper()}")
    print(f"==================================================")

    y_scaled = np.log1p(y) if target_transform == "log1p" else y

    # Doğru tuple açılımı
    (
        X_train,
        X_test,
        y_train,
        y_test,
        w_train,
        w_test,
        y_train_raw,
        y_test_raw,
    ) = train_test_split(
        X, y_scaled, sample_weights, y, test_size=0.2, random_state=42
    )

    print("✂️ Ön Feature Selection uygulanıyor...")
    selector_model = lgb.LGBMRegressor(
        n_estimators=50, random_state=42, verbose=-1
    )
    selector_model.fit(X_train, y_train, sample_weight=w_train)
    selector = SelectFromModel(
        selector_model, threshold="median", prefit=True
    )

    selected_features = X.columns[selector.get_support()].tolist()
    print(
        f"   -> Toplam {X.shape[1]} değişkenden {len(selected_features)} önemli değişken seçildi."
    )

    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    print("🔎 Optuna Hiperparametre Araması Başlatılıyor...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.1, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_sel, y_train, sample_weight=w_train)
        preds = model.predict(X_test_sel)
        return mean_squared_error(y_test, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15, timeout=300)

    best_params = study.best_params
    best_params.update({"random_state": 42, "verbose": -1})

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train_sel, y_train, sample_weight=w_train)

    joblib.dump(final_model, model_save_dir / f"model_{model_name}.pkl")
    joblib.dump(
        (X_test_sel, y_test_raw),
        model_save_dir / f"test_data_{model_name}.pkl",
    )

    print(f"✅ {model_name.upper()} regresyon modeli eğitildi ve kaydedildi!")


# -------------------------------------------------------------------------
# 4. SINIFLANDIRMA MODEL EĞİTİM FONKSİYONU (OWNERS)
# -------------------------------------------------------------------------
def train_and_save_classifier(X, y, model_name, sample_weights=None):
    print(f"\n==================================================")
    print(f"🎯 SINIFLANDIRMA MODELİ EĞİTİLİYOR: {model_name.upper()}")
    print(f"==================================================")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        w_train,
        w_test,
        y_train_raw,
        y_test_raw,
    ) = train_test_split(
        X, y, sample_weights, y, test_size=0.2, random_state=42
    )

    print("✂️ Ön Feature Selection uygulanıyor...")
    selector_model = lgb.LGBMClassifier(
        n_estimators=50, random_state=42, verbose=-1
    )
    selector_model.fit(X_train, y_train, sample_weight=w_train)
    selector = SelectFromModel(
        selector_model, threshold="median", prefit=True
    )

    selected_features = X.columns[selector.get_support()].tolist()
    print(
        f"   -> Toplam {X.shape[1]} değişkenden {len(selected_features)} önemli değişken seçildi."
    )

    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    print("🔎 Optuna Hiperparametre Araması Başlatılıyor...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.1, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "class_weight": "balanced",
            "objective": "multiclass",
            "random_state": 42,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_sel, y_train, sample_weight=w_train)
        preds = model.predict(X_test_sel)
        return f1_score(y_test, preds, average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15, timeout=300)

    best_params = study.best_params
    best_params.update({
        "class_weight": "balanced",
        "objective": "multiclass",
        "random_state": 42,
        "verbose": -1,
    })

    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train_sel, y_train, sample_weight=w_train)

    joblib.dump(final_model, model_save_dir / f"model_{model_name}.pkl")
    joblib.dump(
        (X_test_sel, y_test_raw),
        model_save_dir / f"test_data_{model_name}.pkl",
    )

    print(f"✅ {model_name.upper()} sınıflandırma modeli eğitildi ve kaydedildi!")


# -------------------------------------------------------------------------
# 5. TÜM MODELLERİ ÇAĞIRAN ANA FONKSİYON
# -------------------------------------------------------------------------
def train_all_models():
    # 1. PRICE MODELİ (Uç Değer Filtrelemeli Regresyon)
    df_price = df[(df["price"] >= 0.49) & (df["price"] <= 100.0)].copy()
    X1 = df_price[base_features]
    w1 = weights.loc[df_price.index]

    print(
        f"\n✂️ Price Modeli için uç değerler filtrelendi. Kalan veri satır"
        f" sayısı: {len(df_price)}"
    )
    train_and_save_regressor(
        X1,
        df_price["price"],
        "price",
        target_transform="log1p",
        sample_weights=w1,
    )

    # 2. OWNERS MODELİ (Dahili class_weight='balanced' Aktif)
    X2 = df[base_features + ["price"]]
    train_and_save_classifier(
        X2, df["target_owner_class"], "owners", sample_weights=weights
    )

    # 3. REVIEW MODELİ
    X3 = df[base_features + ["price"]]
    train_and_save_regressor(
        X3,
        df["positive_review_percentage"],
        "review",
        sample_weights=weights,
    )

    print("\n🎉 Tüm optimize edilmiş modeller başarıyla eğitildi!")


# -------------------------------------------------------------------------
# 6. MODEL DEĞERLENDİRME & RAPORLAMA
# -------------------------------------------------------------------------
def evaluate_all_models():
    print("\n" + "=" * 60)
    print("📊 MODEL DEĞERLENDİRME SONUÇLARI")
    print("=" * 60)

    results = {}

    # Price Modeli Evaluation
    if (
        model_save_dir / "model_price.pkl"
    ).exists() and (
        model_save_dir / "test_data_price.pkl"
    ).exists():
        model_p = joblib.load(model_save_dir / "model_price.pkl")
        X_test_p, y_test_real_p = joblib.load(
            model_save_dir / "test_data_price.pkl"
        )

        preds_log_p = model_p.predict(X_test_p)
        preds_real_p = np.expm1(preds_log_p)

        r2_p = r2_score(y_test_real_p, preds_real_p)
        mae_p = mean_absolute_error(y_test_real_p, preds_real_p)
        rmse_p = np.sqrt(mean_squared_error(y_test_real_p, preds_real_p))

        results["price"] = {"r2": r2_p, "mae": mae_p, "rmse": rmse_p}
        print(
            f"💰 PRICE    -> R²: {r2_p:.4f} | MAE: ${mae_p:.2f} | RMSE:"
            f" ${rmse_p:.2f}"
        )

    # Owners Modeli Evaluation
    if (
        model_save_dir / "model_owners.pkl"
    ).exists() and (
        model_save_dir / "test_data_owners.pkl"
    ).exists():
        model_o = joblib.load(model_save_dir / "model_owners.pkl")
        X_test_o, y_test_o = joblib.load(
            model_save_dir / "test_data_owners.pkl"
        )

        preds_o = model_o.predict(X_test_o)

        acc_o = accuracy_score(y_test_o, preds_o)
        f1_o = f1_score(y_test_o, preds_o, average="macro")

        results["owners"] = {"accuracy": acc_o, "macro_f1": f1_o}
        print(f"👥 OWNERS   -> Accuracy: {acc_o:.4f} | Macro F1: {f1_o:.4f}")

    # Review Modeli Evaluation
    if (
        model_save_dir / "model_review.pkl"
    ).exists() and (
        model_save_dir / "test_data_review.pkl"
    ).exists():
        model_r = joblib.load(model_save_dir / "model_review.pkl")
        X_test_r, y_test_r = joblib.load(
            model_save_dir / "test_data_review.pkl"
        )

        preds_r = model_r.predict(X_test_r)

        r2_r = r2_score(y_test_r, preds_r)
        mae_r = mean_absolute_error(y_test_r, preds_r)
        rmse_r = np.sqrt(mean_squared_error(y_test_r, preds_r))

        results["review"] = {"r2": r2_r, "mae": mae_r, "rmse": rmse_r}
        print(
            f"⭐ REVIEW   -> R²: {r2_r:.4f} | MAE: %{mae_r:.2f} puan | RMSE:"
            f" %{rmse_r:.2f} puan"
        )

    print("\n" + "=" * 60)
    print("📌 GENEL METRİK ÖZETİ")
    print("=" * 60)
    df_res = pd.DataFrame(results).T
    print(df_res)


# -------------------------------------------------------------------------
# SCRIPT ÇALIŞTIRMA
# -------------------------------------------------------------------------
if __name__ == "__main__":
    train_all_models()
    evaluate_all_models()