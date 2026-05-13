import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import warnings
warnings.filterwarnings('ignore')

# 1. VERİLERİ YÜKLE
print("Veriler yükleniyor...")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test_x.csv")

# 2. EN İYİ SONUCU VEREN ÖZELLİK MÜHENDİSLİĞİ (Stacking Pro Features)
def create_best_features(df):
    df = df.copy()
    df['hafif_uyku_yuzdesi'] = 100 - (df['rem_yuzdesi'] + df['derin_uyku_yuzdesi'])
    df['kaliteli_uyku_orani'] = (df['rem_yuzdesi'] + df['derin_uyku_yuzdesi']) / (df['hafif_uyku_yuzdesi'] + 1)
    df['uyku_oncesi_zorluk'] = df['uyku_oncesi_ekran_suresi_dk'] * df['uykuya_dalma_suresi_dk']
    
    df['kafein_toleransi'] = df['uyku_oncesi_kafein_mg'] / (df['vucut_kitle_indeksi'] + 1) 
    df['gunluk_hareketlilik'] = df['gunluk_adim_sayisi'] / (df['gunluk_calisma_saati'] + 1)
    df['stres_nabiz_carpimi'] = df['stres_skoru'] * df['dinlenik_nabiz_bpm']
    df['yorgunluk_katsayisi'] = df['yas'] * df['gunluk_calisma_saati']
    return df

train = create_best_features(train)
test = create_best_features(test)

# 3. EKSİK VERİ VE KATEGORİK İŞLEMLER
cat_cols = ['cinsiyet', 'meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi']
for col in cat_cols:
    train[col] = train[col].fillna('Unknown')
    test[col] = test[col].fillna('Unknown')

num_cols = train.select_dtypes(include=['float64', 'int64']).columns.drop(['id', 'bilissel_performans_skoru'], errors='ignore')
for col in num_cols:
    med = train[col].median()
    train[col] = train[col].fillna(med)
    test[col] = test[col].fillna(med)

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train[cat_cols] = oe.fit_transform(train[cat_cols]).astype(int)
test[cat_cols] = oe.transform(test[cat_cols]).astype(int)

X = train.drop(['id', 'bilissel_performans_skoru'], axis=1)
y = train['bilissel_performans_skoru']
test_X = test.drop(['id'], axis=1)

cat_indices = [X.columns.get_loc(c) for c in cat_cols]

# ==========================================
# AŞAMA 1: İLK STACKING (Pseudo-Label Üretimi İçin)
# ==========================================
print("Aşama 1: Pseudo-Labeling için ilk eğitim başlıyor (10 Fold)...")
kf = KFold(n_splits=10, shuffle=True, random_state=42)

meta_train_1 = np.zeros((len(X), 3))
meta_test_lgb_1 = np.zeros(len(test_X))
meta_test_xgb_1 = np.zeros(len(test_X))
meta_test_cb_1 = np.zeros(len(test_X))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # LGBM
    model_lgb = lgb.LGBMRegressor(n_estimators=2500, learning_rate=0.01, num_leaves=45, random_state=42, verbose=-1)
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    meta_train_1[val_idx, 0] = model_lgb.predict(X_val)
    meta_test_lgb_1 += model_lgb.predict(test_X) / kf.n_splits
    
    # XGB
    model_xgb = xgb.XGBRegressor(n_estimators=2500, learning_rate=0.01, max_depth=6, random_state=42, early_stopping_rounds=100)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    meta_train_1[val_idx, 1] = model_xgb.predict(X_val)
    meta_test_xgb_1 += model_xgb.predict(test_X) / kf.n_splits
    
    # CB
    model_cb = cb.CatBoostRegressor(iterations=2500, learning_rate=0.01, depth=7, random_seed=42, early_stopping_rounds=100, verbose=False)
    model_cb.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_indices)
    meta_train_1[val_idx, 2] = model_cb.predict(X_val)
    meta_test_cb_1 += model_cb.predict(test_X) / kf.n_splits

# İlk Stacking Birleşimi
meta_test_1 = np.column_stack([meta_test_lgb_1, meta_test_xgb_1, meta_test_cb_1])
meta_model_1 = Ridge(alpha=1.0)
meta_model_1.fit(meta_train_1, y)
pseudo_labels = meta_model_1.predict(meta_test_1)

# ==========================================
# AŞAMA 2: PSEUDO-LABELING İLE YENİDEN EĞİTİM
# ==========================================
print("\nAşama 2: Test verisi eğitim setine dahil ediliyor! (Pseudo-Labeling)")

X_pseudo = test_X.copy()
y_pseudo = pseudo_labels

X_extended = pd.concat([X, X_pseudo], axis=0).reset_index(drop=True)
y_extended = np.concatenate([y.values, y_pseudo])

# İkinci ve Son Eğitim (Daha Derin, Daha Yavaş)
meta_train_2 = np.zeros((len(X_extended), 3))
meta_test_lgb_2 = np.zeros(len(test_X))
meta_test_xgb_2 = np.zeros(len(test_X))
meta_test_cb_2 = np.zeros(len(test_X))

print("Genişletilmiş Veri İle Son Eğitim Başladı (Bu biraz zaman alabilir)...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_extended, y_extended)):
    X_train, y_train = X_extended.iloc[train_idx], y_extended[train_idx]
    X_val, y_val = X_extended.iloc[val_idx], y_extended[val_idx]
    
    model_lgb = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.005, num_leaves=63, random_state=1903, verbose=-1)
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(150, verbose=False)])
    meta_train_2[val_idx, 0] = model_lgb.predict(X_val)
    meta_test_lgb_2 += model_lgb.predict(test_X) / kf.n_splits
    
    model_xgb = xgb.XGBRegressor(n_estimators=3000, learning_rate=0.005, max_depth=7, random_state=1903, early_stopping_rounds=150)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    meta_train_2[val_idx, 1] = model_xgb.predict(X_val)
    meta_test_xgb_2 += model_xgb.predict(test_X) / kf.n_splits
    
    model_cb = cb.CatBoostRegressor(iterations=3000, learning_rate=0.005, depth=8, random_seed=1903, early_stopping_rounds=150, verbose=False)
    model_cb.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_indices)
    meta_train_2[val_idx, 2] = model_cb.predict(X_val)
    meta_test_cb_2 += model_cb.predict(test_X) / kf.n_splits

# Son Ridge Meta Modeli
meta_test_2 = np.column_stack([meta_test_lgb_2, meta_test_xgb_2, meta_test_cb_2])
meta_model_2 = Ridge(alpha=1.0)
meta_model_2.fit(meta_train_2, y_extended)

final_preds = meta_model_2.predict(meta_test_2)

# Çıktı Alma
submission = pd.DataFrame({
    'id': test['id'],
    'bilissel_performans_skoru': final_preds
})
submission.to_csv('cubmissions/submission_stacking_v2_pseudo.csv', index=False)
print("\nZAFER KODU HAZIR: 'submission_stacking_v2_pseudo.csv' oluşturuldu! Gelsin liderlik tablosu!")