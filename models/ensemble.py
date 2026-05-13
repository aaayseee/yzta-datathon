import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Verileri yükle
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test_x.csv")

# Özellik Mühendisliği (Feature Engineering)
# Verideki değişkenleri kullanarak yeni ufuklar açıyoruz!
def create_features(df):
    df = df.copy()
    
    # Uyku kalitesi ve bozucular
    df['uyku_kalitesi_indeksi'] = df['rem_yuzdesi'] + df['derin_uyku_yuzdesi']
    df['toplam_uyku_bozucu'] = df['uyku_oncesi_kafein_mg'] * df['uyku_oncesi_ekran_suresi_dk']
    
    # Stres ve Yorgunluk (Etkiyi büyütüyoruz)
    df['stres_yuklenmesi'] = df['stres_skoru'] * df['gunluk_calisma_saati']
    df['uykuya_dalma_zorlugu'] = df['uykuya_dalma_suresi_dk'] * df['gecelik_uyanma_sayisi']
    
    return df

train = create_features(train)
test = create_features(test)

# Eksik Değerleri Doldurma
cat_cols = ['cinsiyet', 'meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi']
for col in cat_cols:
    train[col] = train[col].fillna('Unknown')
    test[col] = test[col].fillna('Unknown')

# Sayısal değişkenlerdeki eksikleri medyan ile dolduruyoruz
num_cols = train.select_dtypes(include=['float64', 'int64']).columns.drop(['id', 'bilissel_performans_skoru'], errors='ignore')
for col in num_cols:
    med = train[col].median()
    train[col] = train[col].fillna(med)
    test[col] = test[col].fillna(med)

# Kategorikleri Sayısallaştırma (Ağaç bazlı modeller için)
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train[cat_cols] = oe.fit_transform(train[cat_cols])
test[cat_cols] = oe.transform(test[cat_cols])

# Hedef ve Eğitim Verisini Ayırma
X = train.drop(['id', 'bilissel_performans_skoru'], axis=1)
y = train['bilissel_performans_skoru']
test_X = test.drop(['id'], axis=1)

# 5 K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Modellerin Tahminlerini Tutacağımız Diziler
lgb_oof = np.zeros(len(X))
xgb_oof = np.zeros(len(X))
cb_oof = np.zeros(len(X))

lgb_preds = np.zeros(len(test_X))
xgb_preds = np.zeros(len(test_X))
cb_preds = np.zeros(len(test_X))

print("Eğitim başlıyor... Sıkı tutun!")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # 1. LightGBM Modeli
    model_lgb = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, verbose=-1)
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    lgb_oof[val_idx] = model_lgb.predict(X_val)
    lgb_preds += model_lgb.predict(test_X) / kf.n_splits
    
    # 2. XGBoost Modeli
    model_xgb = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, random_state=42, early_stopping_rounds=50)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_oof[val_idx] = model_xgb.predict(X_val)
    xgb_preds += model_xgb.predict(test_X) / kf.n_splits
    
    # 3. CatBoost Modeli (Arkadaşının modelini daha güçlü parametrelerle eğitiyoruz)
    model_cb = cb.CatBoostRegressor(iterations=1000, learning_rate=0.03, random_seed=42, early_stopping_rounds=50, verbose=False)
    model_cb.fit(X_train, y_train, eval_set=(X_val, y_val))
    cb_oof[val_idx] = model_cb.predict(X_val)
    cb_preds += model_cb.predict(test_X) / kf.n_splits
    
    print(f"Fold {fold+1} tamamlandı.")

# BLENDING: 3 Modelin Ortalamasını Alarak Mükemmeli Yakalama
final_oof = (lgb_oof + xgb_oof + cb_oof) / 3.0
final_preds = (lgb_preds + xgb_preds + cb_preds) / 3.0

print(f"\nFinal Ensemble OOF RMSE: {np.sqrt(mean_squared_error(y, final_oof)):.4f}")

# Sonucu Kaydet
submission = pd.DataFrame({
    'id': test['id'],
    'bilissel_performans_skoru': final_preds
})
submission.to_csv('submissions/submission_ensemble.csv', index=False)
print("Ensemble model başarıyla 'submission_ensemble.csv' olarak kaydedildi!")