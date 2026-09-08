import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    mean_squared_error, 
    accuracy_score, 
    f1_score, 
    classification_report
)

# Dosya yolları
base_dir = Path(__file__).resolve().parent
model_dir = base_dir / "saved_models_xgb"

def evaluate_regressor(model_name):
    print(f"\n==================================================")
    print(f"📊 METRİK RAPORU (REGRESYON): {model_name.upper()}")
    print(f"==================================================")
    
    try:
        model = joblib.load(model_dir / f"model_{model_name}.pkl")
        X_test, y_test = joblib.load(model_dir / f"test_data_{model_name}.pkl")
    except FileNotFoundError:
        print(f"❌ {model_name} için kaydedilmiş model veya test verisi bulunamadı!")
        return

    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"🔹 R² Skoru                : {r2:.4f}")
    print(f"🔹 MAE (Ortalama Mutlak)    : {mae:.2f}")
    print(f"🔹 RMSE (Karakök Ortalama)  : {rmse:.2f}")

def evaluate_classifier(model_name):
    print(f"\n==================================================")
    print(f"📊 METRİK RAPORU (SINIFLANDIRMA): {model_name.upper()}")
    print(f"==================================================")

    try:
        model = joblib.load(model_dir / f"model_{model_name}.pkl")
        X_test, y_test = joblib.load(model_dir / f"test_data_{model_name}.pkl")
        label_encoder = joblib.load(model_dir / f"label_encoder_{model_name}.pkl")
    except FileNotFoundError:
        print(f"❌ {model_name} için kaydedilmiş model veya test verisi bulunamadı!")
        return

    y_pred = model.predict(X_test)
    
    # Label encoder dönüşümü (okunabilir etiketler için)
    target_names = [str(c) for c in label_encoder.classes_]

    acc = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"🔹 Doğruluk (Accuracy)      : %{acc * 100:.2f}")
    print(f"🔹 F1-Score (Weighted)      : {f1_weighted:.4f}")
    print(f"🔹 F1-Score (Macro)         : {f1_macro:.4f}")
    
    print("\n📋 Detaylı Sınıflandırma Raporu:")
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=target_names, 
        zero_division=0
    ))

if __name__ == "__main__":
    # Regresyon Modelleri
    evaluate_regressor("price")
    evaluate_regressor("review")
    
    # Sınıflandırma Modeli
    evaluate_classifier("owners")