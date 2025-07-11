import streamlit as st
import pandas as pd
import pickle

# Load your trained model
@st.cache_resource
def load_model():
    with open("mod.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("🛒 Walmart Customer Churn Prediction App")

st.markdown("""
Welcome to the **Walmart Churn Predictor**.  
Provide some basic customer details, and we’ll predict if the customer is likely to stop shopping at Walmart.
""")

MonthsAsCustomer = st.slider("Months as a Customer", 1, 72, 24)
AvgMonthlySpend = st.number_input("Average Monthly Spend (₹)", 100.0, 10000.0, 500.0)
MembershipType = st.selectbox("Membership Type", ["Month-to-month", "One year", "Two year"])
CustomerSupportUsage = st.selectbox("Customer Support Usage", ["Yes", "No"])
OnlinePreference = st.selectbox("Online Shopping Preference", ["DSL", "Fiber optic", "No"])
PreferredPaymentMethod = st.selectbox("Preferred Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

user_input = pd.DataFrame([{
    'MonthsAsCustomer': MonthsAsCustomer,
    'AvgMonthlySpend': AvgMonthlySpend,
    'MembershipType': MembershipType,
    'CustomerSupportUsage': CustomerSupportUsage,
    'OnlinePreference': OnlinePreference,
    'PreferredPaymentMethod': PreferredPaymentMethod
}])

model_input = user_input.rename(columns={
    'MonthsAsCustomer': 'tenure',
    'AvgMonthlySpend': 'MonthlyCharges',
    'MembershipType': 'Contract',
    'CustomerSupportUsage': 'TechSupport',
    'OnlinePreference': 'InternetService',
    'PreferredPaymentMethod': 'PaymentMethod'
})

if st.button("Predict Churn"):
    prediction = model.predict(model_input)[0]
    if prediction == "Yes":
        st.error("⚠️ This customer is likely to churn!")
    else:
        st.success("✅ This customer is likely to stay.")
