"""
Diyabet Riski Tahmin Uygulamasi - Streamlit
CRISP-DM Metodolojisi ile gelistirilmis Lojistik Regresyon modeli
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Sayfa Ayarlari ---
st.set_page_config(
    page_title="Diyabet Riski Tahmin Sistemi",
    page_icon="🩺",
    layout="centered"
)

# --- Model Yukleme ---
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "diyabet_model.pkl")
    features_path = os.path.join(os.path.dirname(__file__), "feature_names.pkl")
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    return model, feature_names

model, feature_names = load_model()

# --- Yardimci Fonksiyonlar ---
def bmi_category(bmi):
    if bmi <= 18.5:
        return 4  # Underweight
    elif bmi <= 25:
        return 1  # Normal
    elif bmi <= 30:
        return 3  # Overweight
    elif bmi <= 45:
        return 0  # Obese
    else:
        return 2  # Morbid_Obese

def glucose_category(glucose):
    if glucose <= 100:
        return 1  # Normal
    elif glucose <= 125:
        return 2  # Prediabetic
    else:
        return 0  # Diabetic

def age_category(age):
    if age <= 30:
        return 2  # Young
    elif age <= 45:
        return 0  # Middle_Aged
    else:
        return 1  # Senior

# --- Baslik ---
st.markdown("""
# 🩺 Diyabet Riski Tahmin Sistemi
### CRISP-DM Metodolojisi ile Makine Ogrenmesi Projesi

Bu uygulama, **Pima Indians Diabetes** veri seti ile egitilmis bir
**Lojistik Regresyon** modeli kullanarak diyabet riskini tahmin eder.

---
""")

# --- Sidebar: Kullanici Girdileri ---
st.sidebar.header("Hasta Bilgileri")
st.sidebar.markdown("Asagidaki alanlari doldurun:")

pregnancies = st.sidebar.number_input("Hamilelik Sayisi", min_value=0, max_value=20, value=1, step=1)
glucose = st.sidebar.slider("Glikoz (mg/dL)", min_value=0, max_value=200, value=120)
blood_pressure = st.sidebar.slider("Kan Basinci (mm Hg)", min_value=0, max_value=140, value=70)
skin_thickness = st.sidebar.slider("Deri Kalinligi (mm)", min_value=0, max_value=100, value=20)
insulin = st.sidebar.slider("Insulin (mu U/ml)", min_value=0, max_value=850, value=80)
bmi = st.sidebar.slider("BMI (kg/m2)", min_value=0.0, max_value=70.0, value=32.0, step=0.1)
dpf = st.sidebar.slider("Diyabet Pedigree Fonksiyonu", min_value=0.0, max_value=2.5, value=0.47, step=0.01)
age = st.sidebar.slider("Yas", min_value=21, max_value=81, value=33)

# --- Feature Engineering ---
bmi_cat = bmi_category(bmi)
glucose_cat = glucose_category(glucose)
age_cat = age_category(age)

# --- Veriyi Hazirlama ---
input_data = pd.DataFrame({
    "Pregnancies": [pregnancies],
    "Glucose": [float(glucose)],
    "BloodPressure": [float(blood_pressure)],
    "SkinThickness": [float(skin_thickness)],
    "Insulin": [float(insulin)],
    "BMI": [float(bmi)],
    "DiabetesPedigreeFunction": [float(dpf)],
    "Age": [int(age)],
    "BMI_CAT": [bmi_cat],
    "GLUCOSE_CAT": [glucose_cat],
    "AGE_CAT": [age_cat],
})

# Eksik feature varsa sifirla
for col in feature_names:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[feature_names]

# --- Tahmin ---
if st.sidebar.button("Tahmin Yap", type="primary", use_container_width=True):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.markdown("---")
    st.markdown("## Tahmin Sonucu")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Risk Durumu",
            value="RISKLI" if prediction == 1 else "DUSUK RISK"
        )
    with col2:
        st.metric(
            label="Diyabet Olasiligi",
            value=f"%{probability[1]*100:.1f}"
        )
    with col3:
        st.metric(
            label="Saglikli Olasiligi",
            value=f"%{probability[0]*100:.1f}"
        )

    if prediction == 1:
        st.error("**UYARI:** Model, bu bireyde diyabet riski tespit etmistir. Lutfen bir saglik kurulusuna basvurun.")
    else:
        st.success("**SONUC:** Model, bu bireyde dusuk diyabet riski tespit etmistir. Yine de duzenli kontrol onerilir.")

    # Detay tablosu
    st.markdown("### Girilen Degerler")
    display_df = pd.DataFrame({
        "Ozellik": ["Hamilelik", "Glikoz", "Kan Basinci", "Deri Kalinligi",
                     "Insulin", "BMI", "DPF", "Yas"],
        "Deger": [pregnancies, glucose, blood_pressure, skin_thickness,
                  insulin, bmi, dpf, age]
    })
    st.table(display_df)

else:
    st.info("Soldaki panelden hasta bilgilerini girin ve **Tahmin Yap** butonuna basin.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.85em;'>
    Diyabet Riski Tahmin Sistemi | CRISP-DM Metodolojisi | Makine Ogrenmesi Projesi
</div>
""", unsafe_allow_html=True)
