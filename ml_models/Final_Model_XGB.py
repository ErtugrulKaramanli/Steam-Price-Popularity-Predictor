"""
XGBoost karşılaştırma sürümü.

Bu script `Final_Model` (LightGBM) pipeline'ının birebir XGBoost karşılığıdır.
Veri çekme, time-decay ağırlıkları, OOF price üretimi, train/val/test split'leri,
feature selection, Optuna bütçesi ve metrikler aynı tutulmuştur; böylece çıkan
sonuçlar LightGBM sonuçlarıyla doğrudan karşılaştırılabilir.

LightGBM'de birebir karşılığı olmayan noktalar:
  * `num_leaves`        -> XGBoost'ta yok; yerine `min_child_weight` + `reg_lambda`
                           (depthwise büyüme + `max_depth` aynı aralıkta bırakıldı).
  * `class_weight`      -> XGBoost'ta yok; `compute_sample_weight("balanced", y)`
                           ile hesaplanıp time-decay ağırlıklarıyla çarpılıyor.
  * Sınıf etiketleri    -> XGBoost 0..n-1 tamsayı bekler, LabelEncoder eklendi.
  * Feature selection   -> Selector modelde `importance_type="weight"` kullanıldı;
                           LightGBM'in varsayılan "split" önemine denk gelsin diye.

Çıktılar `saved_models_xgb_final/` altına, metrik raporu ise
`ml_model_metrics_report_xgb.csv` dosyasına yazılır (LightGBM çıktıları ezilmez).
"""

import os
from pathlib import Path
import warnings
import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -------------------------------------------------------------------------
# 1. ORTAM VE DİZİN YAPILANMASI
# -------------------------------------------------------------------------
base_dir = Path(__file__).resolve().parent
model_save_dir = base_dir / "saved_models_xgb_final"
model_save_dir.mkdir(parents=True, exist_ok=True)

# Tüm XGBoost modellerinde sabit kalan parametreler
XGB_FIXED_PARAMS = {
    "tree_method": "hist",
    "enable_categorical": True,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

# -------------------------------------------------------------------------
# 2. BIGQUERY VERİ ÇEKME & ÖN İŞLEME
# -------------------------------------------------------------------------
print("🚀 BigQuery bağlantısı kuruluyor ve veriler yükleniyor...")
key_path = base_dir.parent / "bigquery_key.json"
credentials = service_account.Credentials.from_service_account_file(key_path)
client = bigquery.Client(
    credentials=credentials, project="datascientiststeamproject"
)

query = "SELECT * FROM dbt_dataset_steam.ml_features_steam"
df = client.query(query).to_dataframe().reset_index(drop=True)
print(f"✅ Toplam {len(df)} satır veri BigQuery'den çekildi.")

# NOT: Bu liste Final_Model (LightGBM) ile birebir aynı tutuldu.
exclude_cols = [
    "appid",
    "name",
    "release_date",
    "price",
    "positive_review_percentage",
    "target_owner_class",
]
base_features = [c for c in df.columns if c not in exclude_cols]

# Zaman Ağırlıklandırması (Time Decay)
if "release_year" in df.columns:
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(2020)
elif "release_date" in df.columns:
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(2020)
else:
    df["release_year"] = 2020

current_year = 2026
decay_rate = 0.05
weights = pd.Series(np.exp(-decay_rate * (current_year - df["release_year"])), index=df.index)

# XGBoost sayısal olmayan kolonları ancak 'category' tipinde kabul eder
# (enable_categorical=True). LightGBM'de gerekmeyen tek ek ön işleme adımı budur.
categorical_features = []
for col in base_features:
    series = df[col]
    if isinstance(series.dtype, pd.CategoricalDtype):
        categorical_features.append(col)
    elif not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        df[col] = series.astype("category")
        categorical_features.append(col)

if categorical_features:
    print(f"🔤 Kategorik tipe çevrilen kolonlar: {categorical_features}")


# -------------------------------------------------------------------------
# 3. OUT-OF-FOLD (OOF) PRICE ESTIMATION
# -------------------------------------------------------------------------
print("\n🔄 Train/Serving tutarsızlığını önlemek için Out-Of-Fold (OOF) Price tahminleri hesaplanıyor...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_price_preds = np.zeros(len(df))

# Price için geçerli maske
price_valid_mask = (df["price"] >= 0.49) & (df["price"] <= 100.0)

for train_idx, val_idx in kf.split(df):
    # Train indeksi içindeki price filtresine uyan satırların maskelenmesi
    tr_subset = df.iloc[train_idx]
    valid_tr_subset = tr_subset[price_valid_mask.iloc[train_idx]]

    X_tr = valid_tr_subset[base_features]
    y_tr = np.log1p(valid_tr_subset["price"])
    w_tr = weights.loc[valid_tr_subset.index]

    X_val = df.iloc[val_idx][base_features]

    oof_model = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.05, **XGB_FIXED_PARAMS
    )
    oof_model.fit(X_tr, y_tr, sample_weight=w_tr)

    # Validation kümesi için tahmin alıp log1p'den geri çeviriyoruz
    oof_price_preds[val_idx] = np.expm1(oof_model.predict(X_val))

df["predicted_price_oof"] = oof_price_preds
print("✅ OOF Price tahminleri üretildi ve veri setine eklendi.")


# -------------------------------------------------------------------------
# 4. REGRESYON MODEL EĞİTİM FONKSİYONU
# -------------------------------------------------------------------------
def train_and_save_regressor(
    X, y, model_name, target_transform=None, sample_weights=None
):
    print(f"\n==================================================")
    print(f"🎯 REGRESYON MODELİ EĞİTİLİYOR (XGB): {model_name.upper()}")
    print(f"==================================================")

    y_scaled = np.log1p(y) if target_transform == "log1p" else y

    # 1. Aşama: %80 Train+Val, %20 Test
    X_temp, X_test, y_temp, y_test, w_temp, w_test, y_temp_raw, y_test_raw = train_test_split(
        X, y_scaled, sample_weights, y, test_size=0.2, random_state=42
    )

    # 2. Aşama: %80 Temp -> %75 Train (%60 toplam), %25 Val (%20 toplam)
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_temp, y_temp, w_temp, test_size=0.25, random_state=42
    )

    print("✂️ Ön Feature Selection uygulanıyor...")
    # importance_type="weight" -> LightGBM'in varsayılan "split" önemine denk gelir,
    # böylece median eşiği iki modelde de benzer şekilde davranır.
    selector_model = xgb.XGBRegressor(
        n_estimators=50, importance_type="weight", **XGB_FIXED_PARAMS
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
    X_val_sel = X_val[selected_features]
    X_test_sel = X_test[selected_features]

    print("🔎 Optuna Hiperparametre Araması Başlatılıyor (Sadece Validation Seti Üzerinde)...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            **XGB_FIXED_PARAMS,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_sel, y_train, sample_weight=w_train)
        preds_val = model.predict(X_val_sel)
        return mean_squared_error(y_val, preds_val)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=15, timeout=300)

    best_params = study.best_params
    best_params.update(XGB_FIXED_PARAMS)

    # Final Model: Train + Val verisi birleştirilerek eğitilir
    X_full_train = pd.concat([X_train_sel, X_val_sel])
    y_full_train = pd.concat([y_train, y_val])
    w_full_train = pd.concat([w_train, w_val])

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_full_train, y_full_train, sample_weight=w_full_train)

    # Test Kümelerinde Tahmin & Error Margin Hesaplama
    test_preds = final_model.predict(X_test_sel)
    if target_transform == "log1p":
        test_preds_real = np.expm1(test_preds)
    else:
        test_preds_real = test_preds

    residuals = y_test_raw - test_preds_real
    mae_margin = mean_absolute_error(y_test_raw, test_preds_real)
    std_margin = np.std(residuals)

    error_margins = {
        "mae": float(mae_margin),
        "std": float(std_margin),
        "confidence_95": float(1.96 * std_margin)
    }

    bundle = {
        "model": final_model,
        "selected_features": selected_features,
        "best_params": best_params,
        "target_transform": target_transform,
        "error_margins": error_margins
    }

    joblib.dump(bundle, model_save_dir / f"bundle_{model_name}.pkl")
    joblib.dump(
        (X_test_sel, y_test_raw),
        model_save_dir / f"test_data_{model_name}.pkl",
    )

    print(f"✅ {model_name.upper()} regresyon modeli ve paketleri başarıyla kaydedildi!")


# -------------------------------------------------------------------------
# 5. SINIFLANDIRMA MODEL EĞİTİM FONKSİYONU
# -------------------------------------------------------------------------
def balanced_weights(y_fit, base_weights):
    """LightGBM'deki class_weight='balanced' davranışının XGBoost karşılığı:
    sınıf dengeleme ağırlığı ile time-decay ağırlığı çarpılır."""
    class_w = compute_sample_weight(class_weight="balanced", y=y_fit)
    return np.asarray(base_weights, dtype=float) * class_w


def train_and_save_classifier(X, y, model_name, sample_weights=None):
    print(f"\n==================================================")
    print(f"🎯 SINIFLANDIRMA MODELİ EĞİTİLİYOR (XGB): {model_name.upper()}")
    print(f"==================================================")

    # XGBoost 0..n-1 aralığında tamsayı etiket bekler
    label_encoder = LabelEncoder()
    y_encoded = pd.Series(
        label_encoder.fit_transform(y.astype(str)), index=y.index, name=y.name
    )
    num_class = len(label_encoder.classes_)
    print(f"🏷️ Sınıflar ({num_class}): {list(label_encoder.classes_)}")

    # Stratified Split Kullanımı
    X_temp, X_test, y_temp, y_test, w_temp, w_test = train_test_split(
        X, y_encoded, sample_weights, test_size=0.2, random_state=42, stratify=y_encoded
    )

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_temp, y_temp, w_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    w_train_balanced = balanced_weights(y_train, w_train)

    print("✂️ Ön Feature Selection uygulanıyor...")
    selector_model = xgb.XGBClassifier(
        n_estimators=50,
        importance_type="weight",
        objective="multi:softprob",
        num_class=num_class,
        eval_metric="mlogloss",
        **XGB_FIXED_PARAMS,
    )
    selector_model.fit(X_train, y_train, sample_weight=w_train_balanced)
    selector = SelectFromModel(
        selector_model, threshold="median", prefit=True
    )

    selected_features = X.columns[selector.get_support()].tolist()
    print(
        f"   -> Toplam {X.shape[1]} değişkenden {len(selected_features)} önemli değişken seçildi."
    )

    X_train_sel = X_train[selected_features]
    X_val_sel = X_val[selected_features]
    X_test_sel = X_test[selected_features]

    print("🔎 Optuna Hiperparametre Araması Başlatılıyor (Sadece Validation Seti Üzerinde)...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "objective": "multi:softprob",
            "num_class": num_class,
            "eval_metric": "mlogloss",
            **XGB_FIXED_PARAMS,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_sel, y_train, sample_weight=w_train_balanced)
        preds_val = model.predict(X_val_sel)
        return f1_score(y_val, preds_val, average="macro")

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=15, timeout=300)

    best_params = study.best_params
    best_params.update({
        "objective": "multi:softprob",
        "num_class": num_class,
        "eval_metric": "mlogloss",
        **XGB_FIXED_PARAMS,
    })

    # Final Model
    X_full_train = pd.concat([X_train_sel, X_val_sel])
    y_full_train = pd.concat([y_train, y_val])
    w_full_train = pd.concat([w_train, w_val])
    w_full_balanced = balanced_weights(y_full_train, w_full_train)

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_full_train, y_full_train, sample_weight=w_full_balanced)

    # Test seti güven aralıkları
    test_probs = final_model.predict_proba(X_test_sel)
    max_probs = np.max(test_probs, axis=1)

    error_margins = {
        "mean_confidence": float(np.mean(max_probs)),
        "std_confidence": float(np.std(max_probs)),
        "min_confidence": float(np.min(max_probs))
    }

    bundle = {
        "model": final_model,
        "selected_features": selected_features,
        "best_params": best_params,
        "error_margins": error_margins,
        "label_encoder": label_encoder,
        "classes": list(label_encoder.classes_),
    }

    joblib.dump(bundle, model_save_dir / f"bundle_{model_name}.pkl")
    joblib.dump(
        (X_test_sel, y_test),
        model_save_dir / f"test_data_{model_name}.pkl",
    )

    print(f"✅ {model_name.upper()} sınıflandırma modeli ve paketleri başarıyla kaydedildi!")


# -------------------------------------------------------------------------
# 6. TÜM MODELLERİ ÇAĞIRAN ANA FONKSİYON
# -------------------------------------------------------------------------
def train_all_models():
    # 1. PRICE MODELİ
    df_price = df[price_valid_mask].copy()
    X1 = df_price[base_features]
    w1 = weights.loc[df_price.index]

    print(f"\n✂️ Price Modeli için uç değerler filtrelendi. Kalan veri: {len(df_price)}")
    train_and_save_regressor(
        X1,
        df_price["price"],
        "price",
        target_transform="log1p",
        sample_weights=w1,
    )

    # 2. OWNERS MODELİ (OOF Price Kullanılıyor)
    X2 = df[base_features].copy()
    X2["price"] = df["predicted_price_oof"]
    train_and_save_classifier(
        X2, df["target_owner_class"], "owners", sample_weights=weights
    )

    # 3. REVIEW MODELİ (OOF Price Kullanılıyor)
    X3 = df[base_features].copy()
    X3["price"] = df["predicted_price_oof"]
    train_and_save_regressor(
        X3,
        df["positive_review_percentage"],
        "review",
        sample_weights=weights,
    )

    print("\n🎉 Tüm optimize edilmiş XGBoost modelleri başarıyla eğitildi!")


# -------------------------------------------------------------------------
# 7. MODEL DEĞERLENDİRME VE METRİK RAPORLAMA
# -------------------------------------------------------------------------
def evaluate_all_models():
    print("\n" + "=" * 60)
    print("📊 XGBOOST MODEL DEĞERLENDİRME VE METRİK SONUÇLARI")
    print("=" * 60)

    csv_rows = []

    # --- Price Modeli Evaluation ---
    if (model_save_dir / "bundle_price.pkl").exists():
        bundle_p = joblib.load(model_save_dir / "bundle_price.pkl")
        model_p = bundle_p["model"]
        X_test_p, y_test_real_p = joblib.load(model_save_dir / "test_data_price.pkl")

        preds_log_p = model_p.predict(X_test_p)
        preds_real_p = np.expm1(preds_log_p)

        r2_p = r2_score(y_test_real_p, preds_real_p)
        mae_p = mean_absolute_error(y_test_real_p, preds_real_p)
        rmse_p = np.sqrt(mean_squared_error(y_test_real_p, preds_real_p))
        mape_p = np.mean(np.abs((y_test_real_p - preds_real_p) / y_test_real_p)) * 100

        print(f"💰 PRICE MODELİ -> R²: {r2_p:.4f} | MAE: ${mae_p:.2f} | RMSE: ${rmse_p:.2f}")

        csv_rows.append({
            "model": "Price Model (Regressor)",
            "R2": round(r2_p, 4),
            "MAE": round(mae_p, 4),
            "RMSE": round(rmse_p, 4),
            "MAPE_pct": round(mape_p, 2),
            "Accuracy": None,
            "Macro_F1": None,
            "Weighted_F1": None,
            "Precision_Macro": None,
            "Recall_Macro": None,
            "Error_Margin_MAE": round(bundle_p["error_margins"]["mae"], 4),
            "Error_Margin_95_Conf": round(bundle_p["error_margins"]["confidence_95"], 4)
        })

    # --- Owners Modeli Evaluation ---
    if (model_save_dir / "bundle_owners.pkl").exists():
        bundle_o = joblib.load(model_save_dir / "bundle_owners.pkl")
        model_o = bundle_o["model"]
        X_test_o, y_test_o = joblib.load(model_save_dir / "test_data_owners.pkl")

        preds_o = model_o.predict(X_test_o)

        # Metrikler encode edilmiş etiketlerle aynı çıkar, okunabilirlik için geri çeviriyoruz
        label_encoder = bundle_o["label_encoder"]
        y_test_labels = label_encoder.inverse_transform(np.asarray(y_test_o))
        preds_labels = label_encoder.inverse_transform(np.asarray(preds_o))

        acc_o = accuracy_score(y_test_labels, preds_labels)
        f1_macro_o = f1_score(y_test_labels, preds_labels, average="macro")
        f1_weighted_o = f1_score(y_test_labels, preds_labels, average="weighted")
        prec_o = precision_score(y_test_labels, preds_labels, average="macro")
        rec_o = recall_score(y_test_labels, preds_labels, average="macro")

        print(f"👥 OWNERS MODELİ -> Accuracy: {acc_o:.4f} | Macro F1: {f1_macro_o:.4f}")

        csv_rows.append({
            "model": "Owners Model (Classifier)",
            "R2": None,
            "MAE": None,
            "RMSE": None,
            "MAPE_pct": None,
            "Accuracy": round(acc_o, 4),
            "Macro_F1": round(f1_macro_o, 4),
            "Weighted_F1": round(f1_weighted_o, 4),
            "Precision_Macro": round(prec_o, 4),
            "Recall_Macro": round(rec_o, 4),
            "Error_Margin_MAE": None,
            "Error_Margin_95_Conf": None
        })

    # --- Review Modeli Evaluation ---
    if (model_save_dir / "bundle_review.pkl").exists():
        bundle_r = joblib.load(model_save_dir / "bundle_review.pkl")
        model_r = bundle_r["model"]
        X_test_r, y_test_r = joblib.load(model_save_dir / "test_data_review.pkl")

        preds_r = model_r.predict(X_test_r)

        r2_r = r2_score(y_test_r, preds_r)
        mae_r = mean_absolute_error(y_test_r, preds_r)
        rmse_r = np.sqrt(mean_squared_error(y_test_r, preds_r))
        mape_r = np.mean(np.abs((y_test_r - preds_r) / y_test_r)) * 100

        print(f"⭐ REVIEW MODELİ -> R²: {r2_r:.4f} | MAE: %{mae_r:.2f} puan | RMSE: %{rmse_r:.2f} puan")

        csv_rows.append({
            "model": "Review Model (Regressor)",
            "R2": round(r2_r, 4),
            "MAE": round(mae_r, 4),
            "RMSE": round(rmse_r, 4),
            "MAPE_pct": round(mape_r, 2),
            "Accuracy": None,
            "Macro_F1": None,
            "Weighted_F1": None,
            "Precision_Macro": None,
            "Recall_Macro": None,
            "Error_Margin_MAE": round(bundle_r["error_margins"]["mae"], 4),
            "Error_Margin_95_Conf": round(bundle_r["error_margins"]["confidence_95"], 4)
        })

    # Metriklerin CSV'ye kaydedilmesi
    results_df = pd.DataFrame(csv_rows)
    csv_path = base_dir / "ml_model_metrics_report_xgb.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n📂 Tüm detaylı XGBoost metrikleri CSV dosyasına yazıldı: {csv_path}")

    return results_df


# -------------------------------------------------------------------------
# 8. LIGHTGBM İLE KARŞILAŞTIRMA TABLOSU
# -------------------------------------------------------------------------
def compare_with_lightgbm(xgb_results):
    """Final_Model (LightGBM) raporu mevcutsa yan yana karşılaştırma basar."""
    lgbm_csv = base_dir / "ml_model_metrics_report.csv"
    if not lgbm_csv.exists():
        print("\nℹ️ LightGBM metrik raporu bulunamadı, karşılaştırma atlandı.")
        return

    lgbm_results = pd.read_csv(lgbm_csv)

    compare_cols = ["R2", "MAE", "RMSE", "Accuracy", "Macro_F1", "Weighted_F1"]
    merged = lgbm_results.merge(
        xgb_results, on="model", how="outer", suffixes=("_LGBM", "_XGB")
    )

    print("\n" + "=" * 60)
    print("⚔️ LIGHTGBM vs XGBOOST KARŞILAŞTIRMASI")
    print("=" * 60)

    rows = []
    for _, row in merged.iterrows():
        for col in compare_cols:
            lgbm_val = row.get(f"{col}_LGBM")
            xgb_val = row.get(f"{col}_XGB")
            if pd.isna(lgbm_val) and pd.isna(xgb_val):
                continue
            rows.append({
                "model": row["model"],
                "metric": col,
                "LightGBM": lgbm_val,
                "XGBoost": xgb_val,
                "diff (XGB - LGBM)": (
                    round(xgb_val - lgbm_val, 4)
                    if pd.notna(lgbm_val) and pd.notna(xgb_val)
                    else None
                ),
            })

    comparison_df = pd.DataFrame(rows)
    print(comparison_df.to_string(index=False))

    comparison_path = base_dir / "ml_model_metrics_comparison_lgbm_vs_xgb.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n📂 Karşılaştırma tablosu kaydedildi: {comparison_path}")


# -------------------------------------------------------------------------
# SCRIPT ÇALIŞTIRMA
# -------------------------------------------------------------------------
if __name__ == "__main__":
    train_all_models()
    xgb_results = evaluate_all_models()
    compare_with_lightgbm(xgb_results)
