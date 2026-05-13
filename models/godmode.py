import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import HistGradientBoostingRegressor

import warnings
warnings.filterwarnings('ignore')

# 1. VERİ YÜKLEME VE TERSİNE MÜHENDİSLİK HAZIRLIĞI
print("Veriler yükleniyor ve gizli formüller aranıyor...")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test_x.csv")

# 2. DERİN ÖZELLİK MÜHENDİSLİĞİ (Reverse Engineering Features)
def create_god_mode_features(df):
    df = df.copy()
    
    # Doğrusal Olmayan Etkileşimler
    df['stres_x_calisma'] = df['stres_skoru'] * df['gunluk_calisma_saati']
    df['stres_bolu_uyku'] = df['stres_skoru'] / (df['rem_yuzdesi'] + df['derin_uyku_yuzdesi'] + 1)
    df['uyanma_x_dalma'] = df['gecelik_uyanma_sayisi'] * df['uykuya_dalma_suresi_dk']
    df['nabiz_farki'] = df['dinlenik_nabiz_bpm'] - df['stres_skoru']
    df['fiziksel_bitkinlik'] = (df['gunluk_calisma_saati'] * 60) + df['gunluk_adim_sayisi'] / 100
    df['ekran_ve_kafein'] = df['uyku_oncesi_ekran_suresi_dk'] * (df['uyku_oncesi_kafein_mg'] + 1)
    
    # ID Tabanlı Sızıntı (Leakage) Avcısı: İndeks sıralamasını özelliğe çeviriyoruz
    df['id_sirasi'] = df['id'] % 100 
    
    return df

train = create_god_mode_features(train)
test = create_god_mode_features(test)

# 3. KATEGORİK İŞLEMLER VE TARGET ENCODING (Hedef Çıkarımı)
cat_cols = ['cinsiyet', 'meslek', 'ulke', 'kronotip', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi']
for col in cat_cols:
    train[col] = train[col].fillna('Unknown')
    test[col] = test[col].fillna('Unknown')

# Sayısal eksikleri temizle
num_cols = train.select_dtypes(include=['float64', 'int64']).columns.drop(['id', 'bilissel_performans_skoru'], errors='ignore')
for col in num_cols:
    med = train[col].median()
    train[col] = train[col].fillna(med)
    test[col] = test[col].fillna(med)

# Label Encoding
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train[cat_cols] = oe.fit_transform(train[cat_cols]).astype(int)
test[cat_cols] = oe.transform(test[cat_cols]).astype(int)

X = train.drop(['id', 'bilissel_performans_skoru'], axis=1)
y = train['bilissel_performans_skoru'] # Target Transformation
test_X = test.drop(['id'], axis=1)

cat_indices = [X.columns.get_loc(c) for c in cat_cols]

# 4. KAGGLE GRANDMASTER STACKING (10-FOLD CV)
kf = KFold(n_splits=10, shuffle=True, random_state=1903) # Şanslı random state :)

# Meta Model Matrisleri (4 Farklı Model İçin)
meta_train = np.zeros((len(X), 4))
meta_test_lgb = np.zeros(len(test_X))
meta_test_xgb = np.zeros(len(test_X))
meta_test_cb = np.zeros(len(test_X))
meta_test_hist = np.zeros(len(test_X))

print("4'lü Hibrit Stacking Eğitimi Başlıyor! (LGBM + XGB + CatBoost + HistBoost)")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # 1. MODEL: LightGBM (Hızlı ve Derin)
    model_lgb = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.01, num_leaves=63, 
                                  max_depth=8, colsample_bytree=0.65, subsample=0.7, 
                                  random_state=42, verbose=-1)
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(150, verbose=False)])
    meta_train[val_idx, 0] = model_lgb.predict(X_val)
    meta_test_lgb += model_lgb.predict(test_X) / kf.n_splits
    
    # 2. MODEL: XGBoost (Matematiksel ve Keskin)
    model_xgb = xgb.XGBRegressor(n_estimators=3000, learning_rate=0.01, max_depth=7,
                                 colsample_bytree=0.65, subsample=0.7, random_state=42)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    meta_train[val_idx, 1] = model_xgb.predict(X_val)
    meta_test_xgb += model_xgb.predict(test_X) / kf.n_splits
    
    # 3. MODEL: CatBoost (Kategorik Canavarı)
    model_cb = cb.CatBoostRegressor(iterations=3000, learning_rate=0.01, depth=8, 
                                    l2_leaf_reg=4, random_seed=42, verbose=False)
    model_cb.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_indices, early_stopping_rounds=150)
    meta_train[val_idx, 2] = model_cb.predict(X_val)
    meta_test_cb += model_cb.predict(test_X) / kf.n_splits
    
    # 4. MODEL: HistGradientBoosting (Sklearn'in Gizli Silahı)
    model_hist = HistGradientBoostingRegressor(max_iter=1500, learning_rate=0.015, 
                                               max_depth=7, random_state=42, early_stopping=True)
    model_hist.fit(X_train, y_train)
    meta_train[val_idx, 3] = model_hist.predict(X_val)
    meta_test_hist += model_hist.predict(test_X) / kf.n_splits
    
    print(f"Fold {fold+1}/10 Tamamlandı - Veriler harmanlanıyor...")

# 5. META MODEL (RIDGE) - Modelleri Birleştirme Kararı
print("\nMeta Model Eğitiliyor... Sızıntılar ve tahminler birleştiriliyor!")
meta_test = np.column_stack([meta_test_lgb, meta_test_xgb, meta_test_cb, meta_test_hist])

meta_model = Ridge(alpha=5.0)
meta_model.fit(meta_train, y)

# Stacking İçin Performans
stacking_preds = meta_model.predict(meta_train)
stacking_rmse = np.sqrt(mean_squared_error(y, stacking_preds))
print(f"ULTIMATE STACKING OOF RMSE: {stacking_rmse:.5f}")

# 6. SONUÇ ÇIKARTMA (Submission)
final_test_preds = meta_model.predict(meta_test)

submission = pd.DataFrame({
    'id': test['id'],
    'bilissel_performans_skoru': final_test_preds
})
submission.to_csv('submissions/submission_god_mode.csv', index=False)
print("BÜYÜK FİNAL: 'submission_god_mode.csv' başarıyla oluşturuldu! Liderlik tablosu seni bekler!")