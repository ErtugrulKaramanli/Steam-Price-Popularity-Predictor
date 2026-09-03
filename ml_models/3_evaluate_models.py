from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "ml_models" / "saved_models"
PLOT_DIR = BASE_DIR / "ml_models" / "evaluation_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_model(model_name, target_transform=None, unit=""):
    print(f"\n==================================================")
    print(f"📊 METRİK RAPORU: {model_name.upper()}")
    print(f"==================================================")

    model = joblib.load(MODEL_DIR / f"model_{model_name}.pkl")
    X_test, y_test = joblib.load(MODEL_DIR / f"test_data_{model_name}.pkl")

    preds_scaled = model.predict(X_test)

    if target_transform == 'log1p':
        y_test_true = np.expm1(y_test)
        y_preds_true = np.expm1(preds_scaled)
    else:
        y_test_true = y_test
        y_preds_true = preds_scaled

    r2 = r2_score(y_test_true, y_preds_true)
    mae = mean_absolute_error(y_test_true, y_preds_true)
    rmse = np.sqrt(mean_squared_error(y_test_true, y_preds_true))
    
    non_zero_mask = y_test_true > 0
    mape = mean_absolute_percentage_error(y_test_true[non_zero_mask], y_preds_true[non_zero_mask]) * 100

    print(f"🔹 R² Skoru (Açıklayıcılık)      : {r2:.4f}")
    print(f"🔹 MAE (Ortalama Mutlak Hata)    : {mae:,.2f} {unit}")
    print(f"🔹 RMSE (Karakök Ortalama Hata)   : {rmse:,.2f} {unit}")
    print(f"🔹 MAPE (Yüzdesel Ortalama Hata) : %{mape:.2f}")

    # Grafik Çizme
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    clip_val = np.percentile(y_test_true, 98)
    mask = (y_test_true <= clip_val) & (y_preds_true <= clip_val)

    sns.scatterplot(x=y_test_true[mask], y=y_preds_true[mask], alpha=0.3, ax=axes[0], color='#2b5c8f')
    axes[0].plot([0, clip_val], [0, clip_val], 'r--', lw=2, label="Mükemmel Tahmin Çizgisi")
    axes[0].set_title(f"{model_name.capitalize()} - Gerçek vs Tahmin (R²: {r2:.3f})")
    axes[0].set_xlabel(f"Gerçek Değer ({unit})")
    axes[0].set_ylabel(f"Tahmin Edilen ({unit})")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    residuals = y_test_true - y_preds_true
    res_mask = np.abs(residuals) <= np.percentile(np.abs(residuals), 98)
    sns.histplot(residuals[res_mask], kde=True, ax=axes[1], color='#3a923a', bins=30)
    axes[1].axvline(0, color='red', linestyle='--', lw=2)
    axes[1].set_title(f"{model_name.capitalize()} - Hata Dağılımı (MAE: {mae:.2f})")
    axes[1].set_xlabel(f"Hata ({unit})")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"eval_{model_name}.png", dpi=300)
    plt.close()
    print(f"📸 Grafikler oluşturuldu: ml_models/evaluation_plots/eval_{model_name}.png")

# Raporları Çalıştır
evaluate_model("price", target_transform='log1p', unit="$")
evaluate_model("owners", target_transform='log1p', unit="Oyuncu")
evaluate_model("review", unit="%")