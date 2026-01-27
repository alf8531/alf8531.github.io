import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Diabetic Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

# ==========================================
# 2. LOAD SAVED MODEL
# ==========================================
@st.cache_resource
def load_model():
    try:
        # Load the pipeline we saved in the Jupyter Notebook
        model = joblib.load('readmission_final_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Error: Model file 'readmission_final_model.joblib' not found. Please run the training notebook first.")
        return None

pipeline = load_model()

# ==========================================
# 3. USER INTERFACE (SIDEBAR INPUTS)
# ==========================================
st.sidebar.header("Patient Clinical Profile")
st.sidebar.markdown("Adjust parameters to assess risk.")

def user_input_features():
    # --- Demographics ---
    st.sidebar.subheader("Demographics")
    age = st.sidebar.selectbox("Age Group",
        ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
         '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'], index=6)

    gender = st.sidebar.selectbox("Gender", ['Female', 'Male'])

    race = st.sidebar.selectbox("Race",
        ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'])

    # --- Clinical Encounters ---
    st.sidebar.subheader("Encounter Metrics")
    time_in_hospital = st.sidebar.slider("Time in Hospital (Days)", 1, 14, 3)
    num_lab_procedures = st.sidebar.slider("Number of Lab Procedures", 1, 130, 40)

    # --- History & Diagnosis ---
    st.sidebar.subheader("History & Diagnosis")
    number_emergency = st.sidebar.number_input("Emergency Visits (Prior Year)", 0, 100, 0)
    number_inpatient = st.sidebar.number_input("Inpatient Visits (Prior Year)", 0, 100, 0)

    primary_diagnosis = st.sidebar.selectbox("Primary Diagnosis Category",
        ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury',
         'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other'])

    # --- Medication ---
    st.sidebar.subheader("Medication")
    insulin = st.sidebar.selectbox("Insulin Therapy", ['No', 'Steady', 'Up', 'Down'])
    diabetes_med = st.sidebar.selectbox("On Diabetes Meds?", ['Yes', 'No'])

    # Organize features EXACTLY as they were in the training dataframe
    data = {
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'primary_diagnosis_group': primary_diagnosis,
        'race': race,
        'age': age,
        'insulin': insulin,
        'diabetesMed': diabetes_med,
        'gender': gender
    }

    return pd.DataFrame(data, index=[0])

# Get input from user
input_df = user_input_features()

# ==========================================
# 4. MAIN DASHBOARD DISPLAY
# ==========================================
st.title("🏥 Strategic Readmission Risk Framework")
st.markdown("""
This tool utilizes a **Gradient Boosting Machine (XGBoost)** trained on 100,000+ clinical encounters
to forecast the likelihood of a diabetic patient being readmitted to the hospital within 30 days.
""")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Patient Snapshot")
    st.table(input_df)

# ==========================================
# 5. PREDICTION LOGIC
# ==========================================
if pipeline is not None:
    # Button to Trigger Prediction
    if st.button('Assess Readmission Risk'):

        # 1. Get Probability
        # predict_proba returns [prob_class_0, prob_class_1]
        prediction_prob = pipeline.predict_proba(input_df)[0][1]

        # 2. Convert to Percentage
        risk_score = prediction_prob * 100

        with col2:
            st.subheader("Risk Assessment")

            # Dynamic Visuals based on Score
            if risk_score > 50:
                st.error(f"⚠️ HIGH RISK: {risk_score:.1f}%")
                st.markdown("**Recommendation:** Initiate discharge planning protocol C. Schedule follow-up within 72 hours.")
            elif risk_score > 30:
                st.warning(f"⚖️ MODERATE RISK: {risk_score:.1f}%")
                st.markdown("**Recommendation:** Review medication adherence and educate on glycemic control.")
            else:
                st.success(f"✅ LOW RISK: {risk_score:.1f}%")
                st.markdown("**Recommendation:** Standard discharge procedures.")

            # Gauge Chart (Progress Bar)
            st.progress(int(risk_score))
            st.caption("Probability of readmission < 30 days")

        # 3. Explainability (Feature Impact Mockup)
        # Note: Real SHAP values are heavy to calculate in real-time,
        # so we provide context based on the input values.
        st.markdown("---")
        st.subheader("Key Risk Drivers Detected")

        drivers = []
        if input_df['number_inpatient'][0] > 0:
            drivers.append(f"• **Inpatient History:** Patient has {input_df['number_inpatient'][0]} recent inpatient visits.")
        if input_df['number_emergency'][0] > 0:
            drivers.append(f"• **Emergency Utilization:** High utilization of ER services ({input_df['number_emergency'][0]} visits).")
        if input_df['time_in_hospital'][0] > 7:
            drivers.append("• **Length of Stay:** Extended hospitalization duration indicates complexity.")
        if input_df['num_lab_procedures'][0] > 50:
            drivers.append("• **Clinical Complexity:** High volume of lab procedures required.")

        if len(drivers) > 0:
            for d in drivers:
                st.markdown(d)
        else:
            st.markdown("• No specific high-risk flags detected in encounter history.")

else:
    st.warning("Model not loaded. Check file path.")
