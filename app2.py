import streamlit as st
import joblib
import pandas as pd
import os
from datetime import datetime

st.set_page_config(page_title="Morning Star Hospital", page_icon="🏥")

st.markdown("""
<style>
.main { background: linear-gradient(135deg, #fff5f5 0%, #ffffff 100%); }
.stButton > button {
    background-color: #DC2626;
    color: white;
    border-radius: 8px;
    font-weight: 600;
    width: 100%;
}
.stButton > button:hover { background-color: #991B1B; }
h1 { color: #991B1B; }
.card {
    background: white;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    box-shadow: 0 2px 12px rgba(153,27,27,0.08);
    margin-bottom: 1.2rem;
}
.section-title {
    font-weight: 600;
    color: #991B1B;
    border-bottom: 2px solid #FEE2E2;
    padding-bottom: 0.3rem;
    margin-bottom: 0.8rem;
}
.risk-low { background:#ECFDF5; border-left:5px solid #059669; padding:1rem 1.2rem; border-radius:10px; color:#065F46; font-weight:600; font-size:1.05rem; }
.risk-moderate { background:#FFFBEB; border-left:5px solid #D97706; padding:1rem 1.2rem; border-radius:10px; color:#92400E; font-weight:600; font-size:1.05rem; }
.risk-high { background:#FEF2F2; border-left:5px solid #DC2626; padding:1rem 1.2rem; border-radius:10px; color:#7F1D1D; font-weight:600; font-size:1.05rem; }
</style>
""", unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

if st.session_state.logged_in == False:
    st.title("🏥 Morning Star Hospital")
    st.subheader("Diabetes Risk Prediction System")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if (username == "doctor" and password == "adun2026") or (username == "nurse" and password == "nurse2026") or (username == "admin" and password == "admin2026"):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Wrong username or password!")
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.title("🏥 Welcome to Morning Star Hospital")
    st.subheader("Diabetes Risk Prediction System")
    st.success(f"Logged in as: {st.session_state.username}")

    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("diabetes_scaler.pkl")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-title'>👤 Patient Details</p>", unsafe_allow_html=True)
    patient_id = st.text_input("Patient ID", placeholder="e.g. MSH-001")

    col1, col2 = st.columns(2)
    with col1:
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=50.0, max_value=200.0, value=100.0)
        age = st.number_input("Age (years)", min_value=18, max_value=90, value=30)
    with col2:
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=100.0, value=25.0)
        bp = st.number_input("Blood Pressure (mmHg)", min_value=40.0, max_value=130.0, value=70.0)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Predict Diabetes Risk"):
        if not patient_id:
            st.warning("Please enter a Patient ID before predicting.")
        else:
            input_data = pd.DataFrame([[bmi, age, glucose, bp]], columns=scaler.feature_names_in_)
            input_scaled = scaler.transform(input_data)
            prob = model.predict_proba(input_scaled)[0][1] * 100

            if prob < 30:
                risk = "LOW"
                st.markdown(f"<div class='risk-low'>✅ LOW RISK — {prob:.1f}%</div>", unsafe_allow_html=True)
            elif prob < 60:
                risk = "MODERATE"
                st.markdown(f"<div class='risk-moderate'>⚠️ MODERATE RISK — {prob:.1f}%</div>", unsafe_allow_html=True)
            else:
                risk = "HIGH"
                st.markdown(f"<div class='risk-high'>🚨 HIGH RISK — {prob:.1f}%</div>", unsafe_allow_html=True)

            record = {
                "Patient ID": patient_id,
                "Glucose": glucose,
                "BMI": bmi,
                "Age": age,
                "Blood Pressure": bp,
                "Risk Score (%)": round(prob, 1),
                "Risk Level": risk,
                "Date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "Assessed By": st.session_state.username
            }
            csv_path = "patient_records.csv"
            if os.path.exists(csv_path):
                df_existing = pd.read_csv(csv_path)
                df_new = pd.concat([df_existing, pd.DataFrame([record])], ignore_index=True)
            else:
                df_new = pd.DataFrame([record])
            df_new.to_csv(csv_path, index=False)
            st.success(f"Record saved for Patient {patient_id}!")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-title'>📋 Patient Records</p>", unsafe_allow_html=True)
    csv_path = "patient_records.csv"
    if os.path.exists(csv_path):
        df_records = pd.read_csv(csv_path)
        st.dataframe(df_records, use_container_width=True)
        st.download_button("📥 Download Records as CSV", df_records.to_csv(index=False), "patient_records.csv", "text/csv")
    else:
        st.info("No records saved yet.")
    st.markdown("</div>", unsafe_allow_html=True)