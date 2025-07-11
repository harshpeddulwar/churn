import streamlit as st
import joblib
import numpy as np

model = joblib.load("xgbmodel.pkl")

st.title("🛒 Customer Churn Prediction - Walmart Style")
st.markdown("Enter customer details below:")

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 18, 70, 30)
tenure = st.slider("Tenure (in months)", 1, 60, 12)
monthly_spend = st.slider("Avg Monthly Spend (₹)", 500, 15000, 4000)
last_purchase = st.slider("Days Since Last Purchase", 0, 180, 30)
frequency = st.slider("Monthly Purchase Frequency", 0, 20, 3)
category = st.selectbox("Preferred Category", ["Grocery", "Electronics", "Clothing", "Home", "Beauty", "Sports"])
premium = st.selectbox("Is Premium Member?", ["Yes", "No"])
tickets = st.slider("Support Tickets Filed", 0, 10, 1)
satisfaction = st.slider("Satisfaction Score (1-5)", 1, 5, 3)


gender_map = {"Male": 0, "Female": 1, "Other": 2}
category_map = {"Grocery": 0, "Electronics": 1, "Clothing": 2, "Home": 3, "Beauty": 4, "Sports": 5}
premium_map = {"Yes": 1, "No": 0}

input_data = np.array([[
    gender_map[gender],
    age,
    tenure,
    monthly_spend,
    last_purchase,
    frequency,
    category_map[category],
    premium_map[premium],
    tickets,
    satisfaction,
    0, 1, 2, 3, 0, 0  
]])
# Predict
if st.button("🔍 Predict Churn"):
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.error(f"⚠️ This customer is likely to churn. (Probability: {proba:.2f})")
    else:
        st.success(f"✅ This customer is likely to stay. (Probability of churn: {proba:.2f})")
