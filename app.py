import streamlit as st
import pandas as pd
from joblib import load

@st.cache_resource
def load_model():
    return load("modl.pkl")

model = load_model()

st.set_page_config(page_title="Walmart Churn Predictor", layout="centered")
st.title("🛒 Walmart Customer Churn Prediction")

st.markdown("""
Welcome to the **Walmart Customer Churn Prediction App**!  
Fill out the form below to see if a customer might churn.  
This helps identify customers who might stop shopping at Walmart.
""")

MonthsAsCustomer = st.slider("🧍‍♂️ Months as a Customer", 1, 72, 24)
AvgMonthlySpend = st.number_input("💳 Average Monthly Spend (₹)", 100.0, 10000.0, 500.0)
MembershipType = st.selectbox("🎟️ Membership Type", ["Month-to-month", "One year", "Two year"])
CustomerSupportUsage = st.selectbox("📞 Customer Support Usage", ["Yes", "No"])
OnlinePreference = st.selectbox("🛍️ Online Shopping Preference", [
    "Mobile App", "Website", "In-Store Only"
])
PreferredPaymentMethod = st.selectbox("💰 Preferred Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

online_map = {
    "Mobile App": "DSL",
    "Website": "Fiber optic",
    "In-Store Only": "No"
}
mapped_online_preference = online_map[OnlinePreference]

user_input = pd.DataFrame([{
    'MonthsAsCustomer': MonthsAsCustomer,
    'AvgMonthlySpend': AvgMonthlySpend,
    'MembershipType': MembershipType,
    'CustomerSupportUsage': CustomerSupportUsage,
    'OnlinePreference': mapped_online_preference,
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

if st.button("🔮 Predict Churn"):
    prediction = model.predict(model_input)[0]
    if prediction == "Yes":
        st.error("⚠️ This customer is likely to churn! Take retention action.")
    else:
        st.success("✅ This customer is likely to stay loyal to Walmart.")
