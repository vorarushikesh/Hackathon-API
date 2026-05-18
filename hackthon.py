# app.py
# Ultra Modern Diabetes Prediction UI using Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Diabetes Predictor",
    page_icon="🩺",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #0f172a, #111827);
    color: white;
}

.stApp {
    background: linear-gradient(135deg, #0f172a, #111827);
}

.title {
    font-size: 60px;
    font-weight: 800;
    text-align: center;
    color: white;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 20px;
    margin-bottom: 40px;
}

.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 8px 30px rgba(0,0,0,0.3);
}

.result-good {
    background: linear-gradient(90deg,#16a34a,#22c55e);
    padding: 25px;
    border-radius: 20px;
    text-align:center;
    font-size: 30px;
    font-weight: bold;
    color: white;
}

.result-bad {
    background: linear-gradient(90deg,#dc2626,#ef4444);
    padding: 25px;
    border-radius: 20px;
    text-align:center;
    font-size: 30px;
    font-weight: bold;
    color: white;
}

.stButton>button {
    width: 100%;
    height: 60px;
    border-radius: 15px;
    background: linear-gradient(90deg,#3b82f6,#06b6d4);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
    background: linear-gradient(90deg,#2563eb,#0891b2);
}

.metric-box {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = xgb.XGBClassifier()
model.load_model("diabetes_model.ubj")

# ---------------- LOAD FEATURES ----------------
with open("feature_list.json", "r") as f:
    feature_names = json.load(f)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🩺 AI Diabetes Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Advanced Machine Learning Health Risk Analysis System</div>',
    unsafe_allow_html=True
)

# ---------------- LAYOUT ----------------
left, right = st.columns([2,1])

with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📋 Health Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        bmi = st.slider("BMI", 10, 60, 25)
        age = st.slider("Age", 18, 100, 30)
        genhlth = st.slider("General Health", 1, 5, 2)

    with c2:
        highbp = st.selectbox("High BP", [0,1])
        highchol = st.selectbox("High Cholesterol", [0,1])
        smoker = st.selectbox("Smoker", [0,1])

    with c3:
        physactivity = st.selectbox("Physical Activity", [0,1])
        fruits = st.selectbox("Eat Fruits", [0,1])
        veggies = st.selectbox("Eat Veggies", [0,1])

    st.subheader("🧠 Additional Information")

    c4, c5, c6 = st.columns(3)

    with c4:
        stroke = st.selectbox("Stroke", [0,1])
        heart = st.selectbox("Heart Disease", [0,1])
        diffwalk = st.selectbox("Difficulty Walking", [0,1])

    with c5:
        sex = st.selectbox("Sex (0 Female / 1 Male)", [0,1])
        education = st.slider("Education", 1, 6, 4)
        income = st.slider("Income", 1, 8, 5)

    with c6:
        menthlth = st.slider("Mental Health Days", 0, 30, 0)
        physhlth = st.slider("Physical Health Days", 0, 30, 0)
        nodoc = st.selectbox("No Doctor Due Cost", [0,1])

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FEATURE ENGINEERING ----------------
healthscore = genhlth + menthlth + physhlth
bmi_age = bmi * age
comorbidity = highbp + highchol + heart + stroke
lifestyle = physactivity + fruits + veggies
bp_chol = highbp * highchol
age_bmi_risk = age + bmi
health_days_total = menthlth + physhlth
chronic_risk = highbp + highchol + smoker

bmi_category = 0
if bmi < 18.5:
    bmi_category = 0
elif bmi < 25:
    bmi_category = 1
elif bmi < 30:
    bmi_category = 2
else:
    bmi_category = 3

input_data = {
    "HighBP": highbp,
    "HighChol": highchol,
    "CholCheck": 1,
    "BMI": bmi,
    "Smoker": smoker,
    "Stroke": stroke,
    "HeartDiseaseorAttack": heart,
    "PhysActivity": physactivity,
    "Fruits": fruits,
    "Veggies": veggies,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": nodoc,
    "GenHlth": genhlth,
    "MentHlth": menthlth,
    "PhysHlth": physhlth,
    "DiffWalk": diffwalk,
    "Sex": sex,
    "Age": age,
    "Education": education,
    "Income": income,
    "BMI_Age": bmi_age,
    "HealthScore": healthscore,
    "BMI_Category": bmi_category,
    "ComorbidityScore": comorbidity,
    "LifestyleScore": lifestyle,
    "BP_Chol": bp_chol,
    "Age_BMI_Risk": age_bmi_risk,
    "HealthDaysTotal": health_days_total,
    "ChronicRisk": chronic_risk
}

# ---------------- PREDICT BUTTON ----------------
predict = st.button("🚀 Predict Diabetes Risk")

with right:

    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("BMI", bmi)
    st.metric("Age", age)
    st.metric("Lifestyle Score", lifestyle)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if predict:

        input_df = pd.DataFrame([input_data])

        input_df = input_df[feature_names]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        risk = round(max(probability) * 100, 2)

        if prediction == 0:
            st.markdown(
                f'<div class="result-good">✅ LOW DIABETES RISK<br>{risk}% Confidence</div>',
                unsafe_allow_html=True
            )
        elif prediction == 1:
            st.markdown(
                f'<div class="result-bad">⚠️ PRE-DIABETES RISK<br>{risk}% Confidence</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-bad">🚨 HIGH DIABETES RISK<br>{risk}% Confidence</div>',
                unsafe_allow_html=True
            )

        st.progress(int(risk))

        st.subheader("📊 AI Health Insights")

        if bmi > 30:
            st.warning("High BMI detected. Weight management recommended.")

        if physactivity == 0:
            st.warning("Low physical activity detected.")

        if smoker == 1:
            st.warning("Smoking increases diabetes risk.")

        if fruits == 0 or veggies == 0:
            st.warning("Healthy diet improvement recommended.")

# ---------------- FOOTER ----------------
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<center>
<p style='color:gray'>
Built with ❤️ using Streamlit + XGBoost
</p>
</center>
""", unsafe_allow_html=True)