import streamlit as st
import pandas as pd
from joblib import load

@st.cache_resource
def load_model():
    return load("mod.pkl") 

model = load_model()

expected_columns = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Female', 'gender_Male',
    'Partner_No', 'Partner_Yes', 'Dependents_No', 'Dependents_Yes', 'PhoneService_No',
    'PhoneService_Yes', 'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_No', 'PaperlessBilling_Yes',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

st.set_page_config(page_title="🛒 Walmart Churn Predictor", layout="centered")
st.title("🔍 Walmart Customer Churn Prediction")

st.markdown("""
This tool predicts whether a Walmart customer is likely to **churn** (stop shopping).
Fill in the customer details below and hit **Predict Churn** to find out.
""")


with st.form("input_form"):
    gender = st.selectbox("Customer Gender", ['Male', 'Female'])
    SeniorCitizen = st.selectbox("Senior Citizen?", [0, 1])
    Partner = st.selectbox("Has a Registered Partner Account?", ['Yes', 'No'])
    Dependents = st.selectbox("Has Family Dependents?", ['Yes', 'No'])
    tenure = st.slider("Customer Since (Months)", 1, 72, 24)
    
    PhoneService = st.selectbox("Receives Shopping Alerts?", ['Yes', 'No'])
    MultipleLines = st.selectbox("Has Family/Friends Linked?", ['Yes', 'No', 'No phone service'])
    InternetService = st.selectbox("Preferred Shopping Mode", ['Mobile App', 'Website', 'In-Store Only'])
    OnlineSecurity = st.selectbox("Fraud Protection Enabled?", ['Yes', 'No', 'No internet service'])
    OnlineBackup = st.selectbox("Saved Shopping Preferences?", ['Yes', 'No', 'No internet service'])
    DeviceProtection = st.selectbox("Has Walmart+ Mobile Warranty?", ['Yes', 'No', 'No internet service'])
    TechSupport = st.selectbox("Contacted Customer Support?", ['Yes', 'No', 'No internet service'])
    StreamingTV = st.selectbox("Watches Walmart Video Ads?", ['Yes', 'No', 'No internet service'])
    StreamingMovies = st.selectbox("Subscribed to Walmart Entertainment?", ['Yes', 'No', 'No internet service'])
    
    Contract = st.selectbox("Membership Plan", ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.selectbox("Receives e-Receipts?", ['Yes', 'No'])
    PaymentMethod = st.selectbox("Preferred Payment Mode", [
        'Electronic check', 'Mailed check',
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    
    MonthlyCharges = st.number_input("Average Monthly Spend (₹)", min_value=0.0, value=500.0)
    TotalCharges = st.number_input("Total Lifetime Spend (₹)", min_value=0.0, value=2000.0)
    
    submitted = st.form_submit_button("🔮 Predict Churn")


if submitted:
    
    raw_input = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }])

    input_encoded = pd.get_dummies(raw_input)

    for col in expected_columns:
        if col not in input_encoded:
            input_encoded[col] = 0

    input_encoded = input_encoded[expected_columns]

    
    prediction = model.predict(input_encoded)[0]

    if prediction == "Yes":
        st.error("⚠️ This customer is likely to churn! Take action.")
    else:
        st.success("✅ This customer is likely to stay loyal to Walmart.")
