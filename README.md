# Bilişsel Performans Skoru Tahminleme | YTZA Datathon 2026

Bu depo, bireylerin uyku düzeni, yaşam alışkanlıkları ve günlük fizyolojik durum verilerini kullanarak **Bilişsel Performans Skoru** değerini tahmin etmeyi amaçlayan bir regresyon projesini içermektedir. Proje kapsamında; veri ön işleme, derinlemesine özellik mühendisliği (feature engineering) ve ileri seviye topluluk öğrenmesi (ensemble learning) teknikleri kullanılmıştır.

## Proje Klasör Yapısı

Proje, modülerlik ve sürdürülebilirlik ilkelerine uygun olarak aşağıdaki yapıda düzenlenmiştir:

* **`data/`** : Eğitim (`train.csv`) ve test (`test_x.csv`) veri setlerini içerir.
* **`models/`** : Geliştirilen tüm makine öğrenmesi mimarilerini barındırır.
  * `ensemble_base.py`
  * `god_mode_features.py`
  * `stacking_pro.py`
  * `pseudo_labeling_final.py`
* **`submissions/`** : Yarışma (Kaggle) için üretilen final tahmin çıktılarını (`.csv`) barındırır.
* **`README.md`** : Proje dökümantasyonu ve yol haritası.

## Uygulanan Stratejiler ve Modeller

Yarışma sürecinde liderlik tablosunda (Leaderboard) en iyi skoru elde etmek için aşamalı bir mühendislik yol haritası izlenmiştir:

### 1. Baseline & Simple Ensemble
İlk aşamada veri eksiklikleri giderilerek `HistGradientBoosting` ile temel bir tahminleme yapılmış, ardından **LightGBM**, **XGBoost** ve **CatBoost** modellerinin basit ortalamaları alınarak başlangıç referans skoru elde edilmiştir.

### 2. God-Mode (Advanced Feature Engineering)
Verideki gizli örüntüleri (pattern) ve potansiyel veri sızıntılarını (leakage) tespit etmek için geliştirilmiştir:
- **Fizyolojik Etkileşimler:** Kafein toleransı, stres/uyku verimi oranları ve uyku öncesi dijital yorgunluk gibi lineer olmayan yeni değişkenler üretilmiştir.
- **Tersine Mühendislik:** Hedef değişkenin üretim mantığına dair algoritmik çıkarımlar yapılmıştır.

### 3. Stacking Pro
Projenin en güçlü ve istikrarlı ayağıdır. Tek bir modelin kararı yerine, birden fazla modelin tahminlerini öğrenen bir **Meta-Model** yapısı kurulmuştur:
- **Base Models (Temel Modeller):** LightGBM, XGBoost, CatBoost.
- **Meta Model (Üst Karar Verici):** Ridge Regression.
- **Doğrulama (Validation):** 10-Fold Cross-Validation kullanılarak aşırı öğrenme (overfitting) engellenmiştir.

### 4. Pseudo-Labeling Edition (Yarı Denetimli Öğrenme)
Modelin test verisinin dağılımını daha iyi kavrayabilmesi için, Stacking Pro ile elde edilen yüksek doğruluklu test tahminleri (sözde etiketler) ana eğitim setine dahil edilerek model yeniden eğitilmiş ve genelleme kapasitesi artırılmıştır.

## 📊 Geliştirme Süreci ve Sonuçlar

| Aşama | Uygulanan Model / Yöntem | RMSE (OOF) | Durum |
| :--- | :--- | :--- | :--- |
| **v1.0** | HistGradientBoosting (Baseline) | ~1.2314 | Tamamlandı |
| **v2.0** | Simple Ensemble (LGBM + XGB + CB) | ~1.22xx | Tamamlandı |
| **v3.0** | God-Mode Features | ~1.21xx | Tamamlandı |
| **v4.0** | **Stacking Pro (Ridge Meta)** | **1.2051** | **En İyi Temel Skor** |
| **v5.0** | **Stacking Pro + Pseudo-Labeling** | *Hesaplanıyor* | **Final Gönderimi** |

## 🚀 Kurulum ve Kullanım

Projeyi yerel ortamınızda test etmek için aşağıdaki adımları izleyebilirsiniz:

1. Depoyu klonlayın:
   ```bash
   git clone [https://github.com/aaayseee/yzta-datathon.git](https://github.com/aaayseee/yzta-datathon.git)

2. Gerekli Python kütüphanelerini yükleyin:
   ```bash
   pip install pandas numpy scikit-learn lightgbm xgboost catboost

3. Modelleri çalıştırmak için ilgili dosyayı terminal üzerinden başlatın:
   ```bash
   python models/stacking_pro.py
   
