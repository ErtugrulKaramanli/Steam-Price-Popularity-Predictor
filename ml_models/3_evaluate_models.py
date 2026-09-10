import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
)

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
model_save_dir = BASE_DIR / "ml_models" / "saved_models"

if not model_save_dir.exists():
    raise FileNotFoundError(
        f"❌ Model klasörü bulunamadı: {model_save_dir}. Önce 2_train_models.py çalıştırılmalı."
    )


def load_artifacts(model_name):
    model = joblib.load(model_save_dir / f"model_{model_name}.pkl")
    X_test, y_test = joblib.load(model_save_dir / f"test_data_{model_name}.pkl")
    return model, X_test, y_test


# ----------------------------------------------------
# 1. PRICE MODELİ (Regresyon)
# ----------------------------------------------------
def evaluate_price():
    model, X_test, y_test_log = load_artifacts("price")
    preds_log = model.predict(X_test)

    # Log ölçeğinden gerçek Dolar değerlerine dönüşüm
    y_test_real = np.expm1(y_test_log)
    preds_real = np.clip(np.expm1(preds_log), 0, None)

    r2 = r2_score(y_test_real, preds_real)
    mae = mean_absolute_error(y_test_real, preds_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, preds_real))

    print(f"💰 PRICE    -> R²: {r2:.4f} | MAE: ${mae:.2f} | RMSE: ${rmse:.2f}")

    return {
        "model": "price",
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "accuracy": np.nan,
        "macro_f1": np.nan
    }


# ----------------------------------------------------
# 2. OWNERS MODELİ (Sınıflandırma)
# ----------------------------------------------------
def evaluate_owners():
    model, X_test, y_test_enc = load_artifacts("owners")
    preds_enc = model.predict(X_test)

    acc = accuracy_score(y_test_enc, preds_enc)
    macro_f1 = f1_score(y_test_enc, preds_enc, average="macro")

    print(f"👥 OWNERS   -> Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")

    return {
        "model": "owners",
        "r2": np.nan,
        "mae": np.nan,
        "rmse": np.nan,
        "accuracy": acc,
        "macro_f1": macro_f1
    }


# ----------------------------------------------------
# 3. REVIEW MODELİ (Regresyon)
# ----------------------------------------------------
def evaluate_review():
    model, X_test, y_test = load_artifacts("review")
    preds = np.clip(model.predict(X_test), 0, 100)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"⭐ REVIEW   -> R²: {r2:.4f} | MAE: %{mae:.2f} puan | RMSE: %{rmse:.2f} puan")

    return {
        "model": "review",
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "accuracy": np.nan,
        "macro_f1": np.nan
    }


if __name__ == "__main__":
    print("\n📊 MODEL DEĞERLENDİRME SONUÇLARI\n" + "-" * 40)
    
    results = [
        evaluate_price(),
        evaluate_owners(),
        evaluate_review()
    ]

    print("\n" + "=" * 60)
    print("📌 GENEL METRİK ÖZETİ")
    print("=" * 60)
    
    summary_df = pd.DataFrame(results).set_index("model")
    print(summary_df.to_string())

    summary_path = model_save_dir / "evaluation_summary.csv"
    summary_df.to_csv(summary_path)
    print(f"\n💾 Özet tablo kaydedildi: {summary_path}")