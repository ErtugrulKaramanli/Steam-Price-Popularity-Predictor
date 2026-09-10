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
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -------------------------------------------------------------------------
# 1. ORTAM VE DİZİN YAPILANMASI
# -------------------------------------------------------------------------
base_dir = Path(__file__).resolve().parent
model_save_dir = base_dir / "Final_saved_models_XG"
model_save_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 2. BIGQUERY VERİ ÇEKME & ÖN İŞLEME (İki Farklı Tablo)
# -------------------------------------------------------------------------
print("🚀 BigQuery bağlantısı kuruluyor ve veriler yükleniyor...")
key_path = base_dir.parent / "bigquery_key.json"
credentials = service_account.Credentials.from_service_account_file(key_path)
client = bigquery.Client(
    credentials=credentials, project="datascientiststeamproject"
)

# A. Geniş Küme (Price Modeli İçin)
query_all = "SELECT * FROM dbt_dataset_steam.fct_steam_games_ml_all"
df_all = client.query(query_all).to_dataframe().reset_index(drop=True)
print(f"✅ Geniş Küme (Price): {len(df_all)} satır veri çekildi.")

# B. Filtreli Küme (Owners & Review Modelleri İçin)
query_filtered = "SELECT * FROM dbt_dataset_steam.fct_steam_games_ml_filtered_10"
df_filtered = client.query(query_filtered).to_dataframe().reset_index(drop=True)
print(f"✅ Filtreli Küme (Owners & Review): {len(df_filtered)} satır veri çekildi.")

exclude_cols = [
    "app_id",
    "appid",
    "name",
    "release_date",
    "price",
    "positive_review_percentage",
    "target_owner_class",
    "target_review_class",
]

base_features_all = [c for c in df_all.columns if c not in exclude_cols]
base_features_filtered = [c for c in df_filtered.columns if c not in exclude_cols]

# Zaman Ağırlıklandırması (Time Decay)
def calculate_weights(df):
    if "release_year" in df.columns:
        release_years = pd.to_numeric(df["release_year"], errors="coerce").fillna(2020)
    elif "release_date" in df.columns:
        release_years = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(2020)
    else:
        release_years = pd.Series(2020, index=df.index)

    current_year = 2026
    decay_rate = 0.05
    return pd.Series(np.exp(-decay_rate * (current_year - release_years)), index=df.index)

weights_all = calculate_weights(df_all)
weights_filtered = calculate_weights(df_filtered)

# Review Yüzdesini 5 Sınıflı Kategoriye Dönüştürme
def create_review_target_class(pct):
    if pct >= 95:
        return 0  # Overwhelmingly Positive
    elif pct >= 80:
        return 1  # Very Positive / Positive
    elif pct >= 70:
        return 2  # Mostly Positive
    elif pct >= 40:
        return 3  # Mixed
    else:
        return 4  # Negative / Mostly Negative

df_filtered["target_review_class"] = df_filtered["positive_review_percentage"].apply(create_review_target_class)

# -------------------------------------------------------------------------
# 3. OUT-OF-FOLD (OOF) PRICE ESTIMATION (Geniş Küme Üzerinde)
# -------------------------------------------------------------------------
print("\n🔄 Out-Of-Fold (OOF) Price tahminleri hesaplanıyor...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_price_preds = np.zeros(len(df_all))

price_valid_mask = (df_all["price"] >= 0.49) & (df_all["price"] <= 100.0)

for train_idx, val_idx in kf.split(df_all):
    tr_subset = df_all.iloc[train_idx]
    valid_tr_subset = tr_subset[price_valid_mask.iloc[train_idx]]

    X_tr = valid_tr_subset[base_features_all]
    y_tr = np.log1p(valid_tr_subset["price"])
    w_tr = weights_all.loc[valid_tr_subset.index]

    X_val = df_all.iloc[val_idx][base_features_all]

    oof_model = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.05, random_state=42, verbosity=0
    )
    oof_model.fit(X_tr, y_tr, sample_weight=w_tr)

    oof_price_preds[val_idx] = np.expm1(oof_model.predict(X_val))

df_all["predicted_price_oof"] = oof_price_preds

price_oof_map = df_all.set_index("app_id")["predicted_price_oof"].to_dict()
df_filtered["predicted_price_oof"] = df_filtered["app_id"].map(price_oof_map)

print("✅ OOF Price tahminleri üretildi ve iki kümeye de entegre edildi.")

# -------------------------------------------------------------------------
# 4. REGRESYON MODEL EĞİTİM FONKSİYONU
# -------------------------------------------------------------------------
def train_and_save_regressor(
    X, y, df_meta, model_name, target_transform=None, sample_weights=None
):
    print(f"\n==================================================")
    print(f"🎯 REGRESYON MODELİ EĞİTİLİYOR: {model_name.upper()}")
    print(f"==================================================")

    y_scaled = np.log1p(y) if target_transform == "log1p" else y

    (
        X_temp, X_test,
        y_temp, y_test,
        w_temp, w_test,
        y_temp_raw, y_test_raw,
        meta_temp, meta_test
    ) = train_test_split(
        X, y_scaled, sample_weights, y, df_meta[["app_id", "name"]], test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_temp, y_temp, w_temp, test_size=0.25, random_state=42
    )

    print("✂️ Ön Feature Selection uygulanıyor...")
    selector_model = xgb.XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
    selector_model.fit(X_train, y_train, sample_weight=w_train)
    selector = SelectFromModel(selector_model, threshold="median", prefit=True)

    selected_features = X.columns[selector.get_support()].tolist()
    print(f"    -> Toplam {X.shape[1]} değişkenden {len(selected_features)} önemli değişken seçildi.")

    X_train_sel = X_train[selected_features]
    X_val_sel = X_val[selected_features]
    X_test_sel = X_test[selected_features]

    print("🔎 Optuna Hiperparametre Araması Başlatılıyor...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "verbosity": 0,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_sel, y_train, sample_weight=w_train)
        preds_val = model.predict(X_val_sel)
        return mean_squared_error(y_val, preds_val)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=15, timeout=300)

    best_params = study.best_params
    best_params.update({"random_state": 42, "verbosity": 0})

    X_full_train = pd.concat([X_train_sel, X_val_sel])
    y_full_train = pd.concat([y_train, y_val])
    w_full_train = pd.concat([w_train, w_val])

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_full_train, y_full_train, sample_weight=w_full_train)

    test_preds = final_model.predict(X_test_sel)
    test_preds_real = np.expm1(test_preds) if target_transform == "log1p" else test_preds

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
        (X_test_sel, y_test_raw, meta_test),
        model_save_dir / f"test_data_{model_name}.pkl",
    )

    print(f"✅ {model_name.upper()} regresyon modeli kaydedildi!")

# -------------------------------------------------------------------------
# 5. SIRALI REGRESYON (ORDINAL REGRESSION) MODEL EĞİTİM FONKSİYONU
# -------------------------------------------------------------------------
def train_and_save_classifier(X, y, df_meta, model_name, sample_weights=None):
    print(f"\n==================================================")
    print(f"🎯 ORDİNAL REGRESYON MODELİ EĞİTİLİYOR: {model_name.upper()}")
    print(f"==================================================")

    if y.dtype == object or pd.api.types.is_string_dtype(y):
        unique_vals = sorted(y.unique())
        mapping = {val: i for i, val in enumerate(unique_vals)}
        y_numeric = y.map(mapping).astype(float)
        print(f"    -> Etiket haritalaması ({model_name}): {mapping}")
    else:
        y_numeric = y.astype(float)

    (
        X_temp, X_test,
        y_temp, y_test,
        w_temp, w_test,
        meta_temp, meta_test
    ) = train_test_split(
        X, y_numeric, sample_weights, df_meta[["app_id", "name"]],
        test_size=0.2, random_state=42, stratify=y_numeric
    )

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_temp, y_temp, w_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print("✂️ Ön Feature Selection uygulanıyor...")
    selector_model = xgb.XGBRegressor(
        n_estimators=50, random_state=42, verbosity=0
    )
    selector_model.fit(X_train, y_train, sample_weight=w_train)
    selector = SelectFromModel(selector_model, threshold="median", prefit=True)

    selected_features = X.columns[selector.get_support()].tolist()
    print(f"    -> Toplam {X.shape[1]} değişkenden {len(selected_features)} önemli değişken seçildi.")

    X_train_sel = X_train[selected_features]
    X_val_sel = X_val[selected_features]
    X_test_sel = X_test[selected_features]

    print("🔎 Optuna Hiperparametre Araması Başlatılıyor (QWK Amaçlı)...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "objective": "reg:squarederror",
            "random_state": 42,
            "verbosity": 0,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_sel, y_train, sample_weight=w_train)
        preds_val = model.predict(X_val_sel)
        
        preds_rounded = np.clip(np.round(preds_val), y_numeric.min(), y_numeric.max()).astype(int)
        val_actuals = y_val.astype(int)
        
        return cohen_kappa_score(val_actuals, preds_rounded, weights="quadratic")

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=15, timeout=300)

    best_params = study.best_params
    best_params.update({
        "objective": "reg:squarederror",
        "random_state": 42,
        "verbosity": 0,
    })

    X_full_train = pd.concat([X_train_sel, X_val_sel])
    y_full_train = pd.concat([y_train, y_val])
    w_full_train = pd.concat([w_train, w_val])

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_full_train, y_full_train, sample_weight=w_full_train)

    test_preds_continuous = final_model.predict(X_test_sel)

    error_margins = {
        "mean_continuous_pred": float(np.mean(test_preds_continuous)),
        "std_continuous_pred": float(np.std(test_preds_continuous)),
        "min_continuous_pred": float(np.min(test_preds_continuous))
    }

    bundle = {
        "model": final_model,
        "selected_features": selected_features,
        "best_params": best_params,
        "error_margins": error_margins
    }

    joblib.dump(bundle, model_save_dir / f"bundle_{model_name}.pkl")
    joblib.dump(
        (X_test_sel, y_test.astype(int), meta_test),
        model_save_dir / f"test_data_{model_name}.pkl",
    )

    print(f"✅ {model_name.upper()} ordinal regresyon modeli kaydedildi!")

# -------------------------------------------------------------------------
# 6. TÜM MODELLERİ ÇAĞIRAN ANA FONKSİYON
# -------------------------------------------------------------------------
def train_all_models():
    # 1. PRICE MODELİ
    X1 = df_all[base_features_all].copy()
    w1 = weights_all

    print(f"\n✂️ Price Modeli Eğitiliyor (Geniş Küme - Satır: {len(df_all)})")
    train_and_save_regressor(
        X1,
        df_all["price"],
        df_all,
        "price",
        target_transform="log1p",
        sample_weights=w1,
    )

    # 2. OWNERS MODELİ
    X2 = df_filtered[base_features_filtered].copy()
    X2["price"] = df_filtered["predicted_price_oof"]
    w2 = weights_filtered

    print(f"\n✂️ Owners Modeli Eğitiliyor (Filtreli Küme - Satır: {len(df_filtered)})")
    train_and_save_classifier(
        X2, df_filtered["target_owner_class"], df_filtered, "owners", sample_weights=w2
    )

    # 3. REVIEW MODELİ
    X3 = df_filtered[base_features_filtered].copy()
    X3["price"] = df_filtered["predicted_price_oof"]
    w3 = weights_filtered

    print(f"\n✂️ Review Modeli Eğitiliyor (Filtreli Küme - Satır: {len(df_filtered)})")
    train_and_save_classifier(
        X3,
        df_filtered["target_review_class"],
        df_filtered,
        "review",
        sample_weights=w3,
    )

    print("\n🎉 Tüm modeller başarıyla eğitildi!")

# -------------------------------------------------------------------------
# 7. MODEL DEĞERLENDİRME VE METRİK RAPORLAMA
# -------------------------------------------------------------------------
def evaluate_all_models():
    print("\n" + "=" * 60)
    print("📊 MODEL DEĞERLENDİRME VE METRİK SONUÇLARI")
    print("=" * 60)

    csv_rows = []

    # Price Modeli Evaluation
    if (model_save_dir / "bundle_price.pkl").exists():
        bundle_p = joblib.load(model_save_dir / "bundle_price.pkl")
        model_p = bundle_p["model"]
        X_test_p, y_test_real_p, meta_p = joblib.load(model_save_dir / "test_data_price.pkl")

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
            "Quadratic_Kappa": None,
            "Error_Margin_MAE": round(bundle_p["error_margins"]["mae"], 4),
            "Error_Margin_95_Conf": round(bundle_p["error_margins"]["confidence_95"], 4)
        })

    # Owners Modeli Evaluation
    if (model_save_dir / "bundle_owners.pkl").exists():
        bundle_o = joblib.load(model_save_dir / "bundle_owners.pkl")
        model_o = bundle_o["model"]
        X_test_o, y_test_o, meta_o = joblib.load(model_save_dir / "test_data_owners.pkl")

        preds_continuous_o = model_o.predict(X_test_o)
        preds_o = np.clip(np.round(preds_continuous_o), y_test_o.min(), y_test_o.max()).astype(int)

        acc_o = accuracy_score(y_test_o, preds_o)
        qwk_o = cohen_kappa_score(y_test_o, preds_o, weights="quadratic")
        f1_macro_o = f1_score(y_test_o, preds_o, average="macro")
        f1_weighted_o = f1_score(y_test_o, preds_o, average="weighted")
        prec_o = precision_score(y_test_o, preds_o, average="macro", zero_division=0)
        rec_o = recall_score(y_test_o, preds_o, average="macro", zero_division=0)

        print(f"👥 OWNERS MODELİ -> Accuracy: {acc_o:.4f} | QWK: {qwk_o:.4f} | Macro F1: {f1_macro_o:.4f}")

        csv_rows.append({
            "model": "Owners Model (Ordinal Regressor)",
            "R2": None,
            "MAE": None,
            "RMSE": None,
            "MAPE_pct": None,
            "Accuracy": round(acc_o, 4),
            "Macro_F1": round(f1_macro_o, 4),
            "Weighted_F1": round(f1_weighted_o, 4),
            "Precision_Macro": round(prec_o, 4),
            "Recall_Macro": round(rec_o, 4),
            "Quadratic_Kappa": round(qwk_o, 4),
            "Error_Margin_MAE": None,
            "Error_Margin_95_Conf": None
        })

    # Review Modeli Evaluation
    if (model_save_dir / "bundle_review.pkl").exists():
        bundle_r = joblib.load(model_save_dir / "bundle_review.pkl")
        model_r = bundle_r["model"]
        X_test_r, y_test_r, meta_r = joblib.load(model_save_dir / "test_data_review.pkl")

        preds_continuous_r = model_r.predict(X_test_r)
        preds_r = np.clip(np.round(preds_continuous_r), y_test_r.min(), y_test_r.max()).astype(int)

        acc_r = accuracy_score(y_test_r, preds_r)
        qwk_r = cohen_kappa_score(y_test_r, preds_r, weights="quadratic")
        f1_macro_r = f1_score(y_test_r, preds_r, average="macro")
        f1_weighted_r = f1_score(y_test_r, preds_r, average="weighted")
        prec_r = precision_score(y_test_r, preds_r, average="macro", zero_division=0)
        rec_r = recall_score(y_test_r, preds_r, average="macro", zero_division=0)

        print(f"⭐ REVIEW MODELİ -> Accuracy: {acc_r:.4f} | QWK: {qwk_r:.4f} | Macro F1: {f1_macro_r:.4f}")

        csv_rows.append({
            "model": "Review Model (Ordinal Regressor)",
            "R2": None,
            "MAE": None,
            "RMSE": None,
            "MAPE_pct": None,
            "Accuracy": round(acc_r, 4),
            "Macro_F1": round(f1_macro_r, 4),
            "Weighted_F1": round(f1_weighted_r, 4),
            "Precision_Macro": round(prec_r, 4),
            "Recall_Macro": round(rec_r, 4),
            "Quadratic_Kappa": round(qwk_r, 4),
            "Error_Margin_MAE": None,
            "Error_Margin_95_Conf": None
        })

    results_df = pd.DataFrame(csv_rows)
    csv_path = base_dir / "ml_model_metrics_report_XG.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n📂 ML metrikleri CSV dosyasına yazıldı: {csv_path}")

# -------------------------------------------------------------------------
# 8. GÖRSELLEŞTİRME VERİLERİNİ BIGQUERY'E AKTARMA (_XG Tabloları)
# -------------------------------------------------------------------------
def export_viz_data_to_bigquery():
    print("\n" + "=" * 60)
    print("📊 LOOKER STUDIO GÖRSELLEŞTİRME VERİLERİ BİGQUERY'E DÖKÜLÜYOR")
    print("=" * 60)

    dataset_id = "dbt_dataset_steam"

    # A. Regresyon Tahmin Verileri (Price)
    reg_records = []

    if (model_save_dir / "bundle_price.pkl").exists():
        bundle_p = joblib.load(model_save_dir / "bundle_price.pkl")
        X_test_p, y_test_p, meta_p = joblib.load(model_save_dir / "test_data_price.pkl")
        preds_log_p = bundle_p["model"].predict(X_test_p)
        preds_p = np.expm1(preds_log_p)

        for (idx, actual), pred in zip(y_test_p.items(), preds_p):
            app_id_val = meta_p.loc[idx, "app_id"] if "app_id" in meta_p.columns else None
            name_val = meta_p.loc[idx, "name"] if "name" in meta_p.columns else None

            reg_records.append({
                "app_id": app_id_val,
                "name": name_val,
                "model_name": "Price Model",
                "actual_value": float(actual),
                "predicted_value": float(pred),
                "residual": float(actual - pred),
                "abs_error": float(abs(actual - pred)),
            })

    df_reg_viz = pd.DataFrame(reg_records)

    # B. Sınıflandırma/Ordinal Tahmin Verileri (Owners & Review)
    clf_records = []

    for model_name in ["owners", "review"]:
        pkl_path = model_save_dir / f"bundle_{model_name}.pkl"
        if pkl_path.exists():
            bundle = joblib.load(pkl_path)
            X_test, y_test, meta = joblib.load(model_save_dir / f"test_data_{model_name}.pkl")
            preds_continuous = bundle["model"].predict(X_test)
            preds = np.clip(np.round(preds_continuous), y_test.min(), y_test.max()).astype(int)

            for (idx, actual), pred in zip(y_test.items(), preds):
                app_id_val = meta.loc[idx, "app_id"] if "app_id" in meta.columns else None
                name_val = meta.loc[idx, "name"] if "name" in meta.columns else None

                clf_records.append({
                    "app_id": app_id_val,
                    "name": name_val,
                    "model_name": f"{model_name.capitalize()} Model",
                    "actual_class": str(actual),
                    "predicted_class": str(pred),
                    "is_correct": int(actual == pred),
                })

    df_clf_viz = pd.DataFrame(clf_records)

    # C. Feature Importance Verileri
    feat_records = []

    for model_name in ["price", "owners", "review"]:
        pkl_path = model_save_dir / f"bundle_{model_name}.pkl"
        if pkl_path.exists():
            bundle = joblib.load(pkl_path)
            model = bundle["model"]
            features = bundle["selected_features"]
            importances = model.feature_importances_

            for feat, imp in zip(features, importances):
                feat_records.append({
                    "model_name": f"{model_name.capitalize()} Model",
                    "feature_name": feat,
                    "importance_score": float(imp),
                })

    df_feat_viz = pd.DataFrame(feat_records)

    # BigQuery Tablo İsimleri (_XG uzantılı)
    tables = {
        "viz_regression_predictions_XG": df_reg_viz,
        "viz_classification_predictions_XG": df_clf_viz,
        "viz_feature_importances_XG": df_feat_viz,
    }

    for table_name, df_data in tables.items():
        table_ref = f"{client.project}.{dataset_id}.{table_name}"
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")

        job = client.load_table_from_dataframe(df_data, table_ref, job_config=job_config)
        job.result()
        print(f"✅ {table_name} tablosu BigQuery'e yüklendi! (Satır Sayısı: {len(df_data)})")

    print("\n🎉 Tüm görselleştirme tabloları BigQuery'de hazır!")
    

# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------
if __name__ == "__main__":
    train_all_models()
    evaluate_all_models()
    export_viz_data_to_bigquery()
    

    # -------------------------------------------------------------------------
    # 9. METRİK RAPORUNU BİGQUERY'E AKTARMA (viz_model_metrics_XG)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 MODEL METRİKLERİ RAPORU BİGQUERY'E AKTARILIYOR")
    print("=" * 60)

    dataset_id = "dbt_dataset_steam"
    table_name = "viz_model_metrics_XG"
    table_ref = f"{client.project}.{dataset_id}.{table_name}"

    csv_path = base_dir / "ml_model_metrics_report_XG.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Metrik raporu bulunamadı: {csv_path}. Önce metrik raporunun oluşturulduğundan emin olun.")

    df_metrics = pd.read_csv(csv_path)

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(df_metrics, table_ref, job_config=job_config)
    job.result()

    print(f"🎉 `{table_name}` tablosu BigQuery'e başarıyla yüklendi! (Satır Sayısı: {len(df_metrics)})")