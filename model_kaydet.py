"""
Diyabet Tahmin Modeli - Model Eğitme ve Kaydetme Scripti
Bu script, notebook'taki pipeline'ı tekrar oluşturup modeli .pkl olarak kaydeder.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# 1. Veri yükleme
df_raw = pd.read_csv('diabetes.csv')
df = df_raw.copy()

# 2. Gizli eksik değerleri (0'lar) medyan ile doldurma
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# 3. Aykırı değer yönetimi (Winsorization, IQR=3.5)
def detect_outliers_iqr_bounds(data, iqr_multiplier=3.5):
    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
    numeric_cols = [c for c in numeric_cols if c != "Outcome"]
    bounds = {}
    for col in numeric_cols:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        bounds[col] = (lower, upper)
    return bounds

bounds = detect_outliers_iqr_bounds(df, iqr_multiplier=3.5)
df_cleaned = df.copy()
for col, (lower, upper) in bounds.items():
    df_cleaned[col] = df_cleaned[col].clip(lower, upper)
df = df_cleaned.copy()

# 4. Özellik Mühendisliği
bmi_bins = [0, 18.5, 25, 30, 45, df['BMI'].max()]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese', 'Morbid_Obese']
df['BMI_CAT'] = pd.cut(df['BMI'], bins=bmi_bins, labels=bmi_labels)

glucose_bins = [0, 100, 125, df['Glucose'].max()]
glucose_labels = ['Normal', 'Prediabetic', 'Diabetic']
df['GLUCOSE_CAT'] = pd.cut(df['Glucose'], bins=glucose_bins, labels=glucose_labels)

age_bins = [0, 30, 45, df['Age'].max()]
age_labels = ['Young', 'Middle_Aged', 'Senior']
df['AGE_CAT'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

# 5. Label Encoding
le = LabelEncoder()
cat_cols = ['BMI_CAT', 'GLUCOSE_CAT', 'AGE_CAT']
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# 6. Model Eğitimi
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=5000, class_weight="balanced"))
])

model.fit(X_train, y_train)

# 7. Performans
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("=" * 50)
print("MODEL PERFORMANSI")
print("=" * 50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
print("=" * 50)

# 8. Modeli ve feature isimlerini kaydetme
joblib.dump(model, 'diyabet_model.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')

print("\n[OK] Model 'diyabet_model.pkl' olarak kaydedildi.")
print("[OK] Feature isimleri 'feature_names.pkl' olarak kaydedildi.")
