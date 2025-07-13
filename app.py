import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# App Config
st.set_page_config(page_title="💳 Loan Fraud Detection", layout="centered")
st.markdown(
    "<h1 style='text-align: center; color: black; background: linear-gradient(to right, #4A00E0, #8E2DE2); padding: 10px; border-radius: 8px;'>💳 Smart Loan Fraud Detection System</h1>",
    unsafe_allow_html=True
)
st.markdown("### 🧠 Predict whether a loan application is **fraudulent** or **legit** based on AI analysis of financial data.")

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

with st.form("loan_form"):
    st.markdown("---")
    st.subheader("📋 Applicant Details")

    gender = st.selectbox("👤 Gender", ["Male", "Female"])
    married = st.selectbox("💍 Married", ["Yes", "No"])
    dependents = st.selectbox("👶 Number of Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("🎓 Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("🏢 Self Employed", ["Yes", "No"])

    st.markdown("---")
    st.subheader("💰 Financial Information")

    applicant_income = st.number_input("💸 Applicant Income (in ₹ 1000s)", min_value=0)
    coapplicant_income = st.number_input("👥 Co-applicant Income (in ₹ 1000s)", min_value=0)
    loan_amount = st.number_input("🏦 Loan Amount (in ₹ 1000s)", min_value=0)
    loan_term_months = st.number_input("🕐 Loan Term (in Months)", min_value=0)
    credit_history = st.selectbox("🧾 Credit History (1 = Good, 0 = Bad)", [1, 0])
    property_area = st.selectbox("📍 Property Area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("🔍 Predict")

# When form submitted
if submitted:
    # Encoding input values
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    dependents = {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents]
    property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

    # Prepare input array
    data = np.array([[gender, married, dependents, education, self_employed,
                      applicant_income, coapplicant_income, loan_amount,
                      loan_term_months, credit_history, property_area]])

    # Scale features
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]  # legit probability

    st.markdown("---")
    st.subheader("📊 Prediction Summary")

    # Gauge Chart
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(probability * 100, 2),
        title={'text': "Loan Approval Probability (%)"},
        delta={'reference': 70},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green" if prediction == 1 else "crimson"},
            'steps': [
                {'range': [0, 50], 'color': 'lightcoral'},
                {'range': [50, 80], 'color': 'lightyellow'},
                {'range': [80, 100], 'color': 'lightgreen'},
            ],
        }
    ))
    st.plotly_chart(gauge_fig, use_container_width=True)

    # Output Message
    if prediction == 1:
        st.success("✅ **Approved**: The loan is predicted to be *Legit* based on current financials.")
    else:
        st.error("❌ **Fraudulent/Risky**: The loan application is flagged as high risk.")

    # Input Summary
    st.subheader("📌 Input Summary")
    st.markdown(f"""
    - 👤 **Gender**: {'Male' if gender else 'Female'}  
    - 💍 **Married**: {'Yes' if married else 'No'}  
    - 🎓 **Education**: {'Graduate' if education else 'Not Graduate'}  
    - 👶 **Dependents**: {dependents}  
    - 💼 **Self Employed**: {'Yes' if self_employed else 'No'}  
    - 💸 **Applicant Income**: ₹{applicant_income * 1000:,.0f}  
    - 👥 **Co-applicant Income**: ₹{coapplicant_income * 1000:,.0f}  
    - 🏦 **Loan Amount**: ₹{loan_amount * 1000:,.0f}  
    - 🕐 **Loan Term**: {loan_term_months} months  
    - 🧾 **Credit History**: {'Good' if credit_history else 'Bad'}  
    - 📍 **Property Area**: {['Rural', 'Semiurban', 'Urban'][property_area]}  
    """)

    # Explain reason
    st.subheader("🧠 Why this prediction?")
    reasons = []
    if credit_history == 0:
        reasons.append("❌ Bad credit history is a major red flag.")
    if applicant_income * 1000 < 30000:
        reasons.append("💸 Low applicant income may impact repayment.")
    if loan_amount * 1000 > 150000:
        reasons.append("📈 High loan amount requested increases financial risk.")
    if loan_term_months < 60:
        reasons.append("🕐 Short loan term could indicate repayment pressure.")
    if self_employed == 1 and applicant_income * 1000 < 40000 and loan_amount * 1000 > 100000:
        reasons.append("🏢 Self-employed with low income and large loan — high-risk profile.")
    if dependents > 2:
        reasons.append("👪 Many dependents may increase financial strain.")

    if not reasons:
        st.success("✅ All key indicators look healthy. Low fraud risk.")
    else:
        for r in reasons:
            st.markdown(f"- {r}")

    # Fraud Solutions
    if prediction == 0:
        st.markdown("---")
        st.subheader("🛠️ Suggested Improvements")
        st.info("""
        💡 To increase loan approval chances:
        - ✅ Maintain a **good credit score**
        - 📉 Reduce requested **loan amount**
        - 👥 Add a **co-applicant** or guarantor
        - 🕒 Choose a **longer repayment term** to lower EMIs
        - 💼 Increase or document **steady income** (esp. if self-employed)
        - 💳 Pay off previous dues and avoid defaults
        """)
