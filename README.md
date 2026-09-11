# 🎮 STEAMCAST: Steam Oyun Veri Analitiği, Fiyatlandırma ve Pazar Simülatörü

<p align="center">
  <img src="docs/9.jpg" alt="SteamCast Canlı Demo" width="100%">
</p>

<p align="center">
  <b>Bağımsız oyun geliştiricileri için sezgisel kararları veri odaklı stratejilere dönüştüren, uçtan uca makine öğrenmesi ve karar destek sistemi.</b>
</p>

<p align="center">
  <a href="https://steamcast.streamlit.app/" target="_blank">🌐 Canlı Demoyu İncele</a> •
  <a href="#-proje-mimarisi-ve-teknoloji-yığını">⚙️ Mimari</a> •
  <a href="#-proje-dizin-yapısı">📁 Dizin Yapısı</a> •
  <a href="#-makine-öğrenmesi-ve-validasyon-stratejisi">🤖 Modeller</a> •
  <a href="#-bulgular-ve-başarı-analizi">📊 Bulgular</a>
</p>

---

## 🚀 Proje Hakkında

Aşırı doymuş Steam pazarında indie (bağımsız) geliştiricilerin yaşadığı belirsizlikleri ortadan kaldırmak amacıyla geliştirilen **STEAMCAST**, oyunların henüz geliştirme aşamasındayken satış, fiyatlandırma, oyuncu sayısı ve kullanıcı memnuniyetini simüle etmesini sağlayan bir karar destek sistemidir.

Proje, Kaggle'dan alınan 139k+ oyunluk ham veritabanının BigQuery ve dbt ile temizlenmesi, LightGBM ve XGBoost modelleriyle eğitilmesi ve Streamlit tabanlı interaktif bir arayüzle canlıya alınması süreçlerini kapsayan uçtan uca bir veri bilimi pipeline'ı sunar.

---

## 📈 Pazar Analizi ve Problem Tanımı

Valve Corporation tarafından geliştirilen Steam, PC oyun pazarının %80'ini domine etmektedir. Ancak her yıl binlerce yeni oyunun yayınlanması pazardaki rekabeti zirveye taşımıştır:

* **Fiyatlandırma Çıkmazı:** Oyunların büyük çoğunluğu dar fiyat bantlarına ($0-$10$) sıkışmıştır; yanlış fiyatlandırma doğrudan geliri ve rekabet şansını düşürür.
* **Kitle ve Satış Belirsizliği:** Bağımsız geliştiriciler, veri odaklı pazar araştırması bütçelerine sahip olmadıkları için kritik kararları sezgisel almak zorunda kalmaktadır.
* **İnceleme Skoru Endişesi:** Görünürlük için kritik olan %70+ pozitif skor barajını yakalamak doğru tür stratejisi gerektirir.

<p align="center">
  <img src="docs/1.jpg" alt="Steam Ekosistemi" width="48%">
  <img src="docs/2.jpg" alt="Problem Tanımı" width="48%">
</p>

---

## 🛠️ Proje Mimarisi ve Teknoloji Yığını

Projede modern veri mühendisliği ve makine öğrenmesi araçları entegre bir şekilde çalışmaktadır:

| Aşama | Kullanılan Teknolojiler / Araçlar | Açıklama |
| :--- | :--- | :--- |
| **Veri Kaynağı & Depolama** | Kaggle, Google BigQuery | 139k+ oyunluk ham veri seti BigQuery üzerinde depolandı ve modellendi. |
| **Veri Mühendisliği & Pipeline** | dbt (data build tool), Python | Eksik/ölü kayıtlar temizlendi, Staging ve Mart katmanları oluşturuldu, Jinja ile Multi-Hot Encoding uygulandı. |
| **Makine Öğrenmesi** | LightGBM, XGBoost, Optuna | Regresyon ve sınıflandırma modelleri, zamansal ağırlıklandırma ve OOF (Out-of-Fold) yaklaşımlarıyla eğitildi. |
| **Açıklanabilirlik & Arayüz** | Streamlit, SHAP | Geliştiricilerin anlık simülasyon yapabileceği, SHAP grafikleriyle desteklenen interaktif web arayüzü. |

<p align="center">
  <img src="docs/3.jpg" alt="Çözüm Mimarisi" width="100%">
</p>

---

## 📁 Proje Dizin Yapısı

Projenin modüler ve katmanlı dosya yapısı aşağıdaki gibidir:

```text
.
├── analyses/                  # Ad-hoc SQL analizleri ve sorgular
├── data/                      # İşlenmiş veri setleri (Parquet ve CSV formatlarında)
├── docs/                      # Sunum PDF dosyası ve ekran görüntüleri (1.jpg - 9.jpg)
├── macros/                    # dbt makroları ve özel fonksiyonlar
├── ml_models/                 # Makine öğrenmesi kodları, pipeline scriptleri ve kaydedilmiş model dosyaları
│   ├── evaluation_plots/      # Model başarı ve hata değerlendirme grafikleri
│   └── saved_models/          # Eğitilmiş final model dosyaları (.pkl / binary)
├── models/                    # dbt model katmanları
│   ├── marts/                 # İş zekası ve ML model girdileri için nihai tablolar (fct_*)
│   └── staging/               # Ham verileri temizleyen ilk katman modelleri (stg_*)
├── seeds/                     # dbt başlangıç veri setleri
├── snapshots/                 # dbt yavaş değişen boyut (SCD) snapshot konfigürasyonları
├── tests/                     # dbt veri kalitesi testleri
├── .gitignore                 # Git ignore kuralları
├── dbt_project.yml            # dbt proje konfigürasyon dosyası
└── README.md                  # Proje dokümantasyonu