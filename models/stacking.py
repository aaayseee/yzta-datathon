import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# 1. Verileri Yükle
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test_x.csv")

# 2. Üst Düzey Özellik Mühendisliği (Grandmaster Features)
def create_advanced_features(df):
    df = df.copy()
    
    # Uyku Dinamikleri
    df['hafif_uyku_yuzdesi'] = 100 - (df['rem_yuzdesi'] + df['derin_uyku_yuzdesi'])
    df['kaliteli_uyku_orani'] = (df['rem_yuzdesi'] + df['derin_uyku_yuzdesi']) / (df['hafif_uyku_yuzdesi'] + 1)
    df['uyku_oncesi_zorluk'] = df['uyku_oncesi_ekran_suresi_dk'] * df['uykuya_dalma_suresi_dk']
    
    # Fizyolojik ve Yaşam Tarzı Etkileşimleri
    df['kafein_toleransi'] = df['uyku_oncesi_kafein_mg'] / (df['vucut_kitle_indeksi'] + 1) # BMI yüksekse kafein etkisi azalabilir
    df['gunluk_hareketlilik'] = df['gunluk_adim_sayisi'] / (df['gunluk_calisma_saati'] + 1)
    df['stres_nabiz_carpimi'] = df['stres_skoru'] * df['dinlenik_nabiz_bpm']
    
    # Yaş ve Çalışma Dinamikleri
    df['yorgunluk_katsayisi'] = df['yas'] * df['gunluk_calisma_saati']
    
    return df

train = create_advanced_features(train)
test = create_advanced_features(test)

# 3. Eksik Veri Yönetimi
cat_cols = ['cinsiyet', 'meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi']
for col in cat_cols:
    train[col] = train[col].fillna('Unknown')
    test[col] = test[col].fillna('Unknown')

num_cols = train.select_dtypes(include=['float64', 'int64']).columns.drop(['id', 'bilissel_performans_skoru'], errors='ignore')
for col in num_cols:
    med = train[col].median()
    train[col] = train[col].fillna(med)
    test[col] = test[col].fillna(med)

# 4. Kategorik Değişkenleri Sayısallaştırma
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train[cat_cols] = oe.fit_transform(train[cat_cols]).astype(int)
test[cat_cols] = oe.transform(test[cat_cols]).astype(int)

X = train.drop(['id', 'bilissel_performans_skoru'], axis=1)
y = train['bilissel_performans_skoru']
test_X = test.drop(['id'], axis=1)

cat_indices = [X.columns.get_loc(c) for c in cat_cols]

# 5. Stacking Mimarisi Hazırlığı (10-Fold CV)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Meta Model (Ridge) için OOF (Out-of-Fold) tahminlerini tutacağımız matrisler
meta_train = np.zeros((len(X), 3))
meta_test_lgb = np.zeros(len(test_X))
meta_test_xgb = np.zeros(len(test_X))
meta_test_cb = np.zeros(len(test_X))

print("Kaggle Stacking Mimarisi Eğitimi Başlıyor! (10 Fold)")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # --- MODEL 1: LightGBM ---
    model_lgb = lgb.LGBMRegressor(
        n_estimators=2500, learning_rate=0.01, num_leaves=45, 
        colsample_bytree=0.7, subsample=0.7, random_state=42, verbose=-1
    )
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    meta_train[val_idx, 0] = model_lgb.predict(X_val)
    meta_test_lgb += model_lgb.predict(test_X) / kf.n_splits
    
    # --- MODEL 2: XGBoost ---
    model_xgb = xgb.XGBRegressor(
        n_estimators=2500, learning_rate=0.01, max_depth=6,
        colsample_bytree=0.7, subsample=0.7, random_state=42, early_stopping_rounds=100
    )
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    meta_train[val_idx, 1] = model_xgb.predict(X_val)
    meta_test_xgb += model_xgb.predict(test_X) / kf.n_splits
    
    # --- MODEL 3: CatBoost ---
    model_cb = cb.CatBoostRegressor(
        iterations=2500, learning_rate=0.01, depth=7, l2_leaf_reg=3,
        random_seed=42, early_stopping_rounds=100, verbose=False
    )
    # CatBoost'a kategorik değişkenleri özel olarak belirtiyoruz (Skoru en çok zıplatan detaylardan biri)
    model_cb.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_indices)
    meta_train[val_idx, 2] = model_cb.predict(X_val)
    meta_test_cb += model_cb.predict(test_X) / kf.n_splits
    
    print(f"Fold {fold+1} tamamlandı.")

# 6. Meta Model (Ridge Regresyon) ile Stacking
print("\nMeta Model (Ridge) Eğitiliyor...")
# Test seti için tahminleri birleştiriyoruz
meta_test = np.column_stack([meta_test_lgb, meta_test_xgb, meta_test_cb])

# 3 modelin ürettiği tahminleri yeni "özellikler" (feature) olarak kullanıp asıl hedefi tahmin ediyoruz
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_train, y)

# Stacking modelimizin nihai OOF hatası
final_oof_preds = meta_model.predict(meta_train)
stacking_rmse = np.sqrt(mean_squared_error(y, final_oof_preds))
print(f"BOMBA GİBİ Stacking OOF RMSE: {stacking_rmse:.4f}")

# 7. Nihai Submission Dosyası
final_test_preds = meta_model.predict(meta_test)

submission = pd.DataFrame({
    'id': test['id'],
    'bilissel_performans_skoru': final_test_preds
})
submission.to_csv('submissions/submission_stacking_pro.csv', index=False)
print("Şampiyonluk modeli 'submission_stacking_pro.csv' dosyasına kaydedildi!")