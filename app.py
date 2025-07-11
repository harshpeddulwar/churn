import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title("🛒 Customer Churn Prediction - Walmart Style")
st.markdown("Enter customer details below:")

model = joblib.load("xgb_churn_model.pkl")
feature_columns = joblib.load("features.pkl")


gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 18, 70, 30)
city_tier = st.selectbox("City Tier", [1, 2, 3])
tenure = st.slider("Tenure (months)", 1, 60, 12)
monthly_spend = st.slider("Avg Monthly Spend (₹)", 500, 15000, 4000)
last_purchase = st.slider("Days Since Last Purchase", 0, 180, 30)
frequency = st.slider("Monthly Purchase Frequency", 1, 20, 3)
premium = st.selectbox("Is Premium Member?", ["Yes", "No"])
online = st.selectbox("Prefers Online Shopping?", ["Yes", "No"])
tickets = st.slider("Support Tickets", 0, 10, 1)
satisfaction = st.slider("Satisfaction Score (1-5)", 1, 5, 3)
category = st.selectbox("Preferred Category", ["Grocery", "Electronics", "Clothing", "Home", "Beauty", "Sports"])

gender_map = {"Male": 0, "Female": 1, "Other": 2}
category_map = {"Grocery": 0, "Electronics": 1, "Clothing": 2, "Home": 3, "Beauty": 4, "Sports": 5}
premium_map = {"Yes": 1, "No": 0}
online_map = {"Yes": 1, "No": 0}

promo_response = 0.3
seasonal_score = 0.6

lifetime_value = monthly_spend * tenure
avg_order_value = monthly_spend / frequency
rfm_score = 5 + 1 + 1 
spend_per_visit = monthly_spend / frequency
tenure_per_age = tenure / age
spend_per_tenure = monthly_spend / tenure
satisfaction_per_ticket = satisfaction / (tickets + 1)
engagement_score = promo_response * 0.3 + seasonal_score * 0.3 + (satisfaction / 5) * 0.4
high_value = int(monthly_spend > 6000 or premium_map[premium] == 1)
high_risk = int(last_purchase > 60 or satisfaction <= 2 or tickets >= 2)
customer_segment = 2 

input_dict = {
    'Age': age,
    'CityTier': city_tier,
    'Tenure': tenure,
    'AvgMonthlySpend': monthly_spend,
    'LastPurchaseDaysAgo': last_purchase,
    'PurchaseFrequency': frequency,
    'IsPremiumMember': premium_map[premium],
    'OnlinePreference': online_map[online],
    'SupportTickets': tickets,
    'SatisfactionScore': satisfaction,
    'PromoResponseRate': promo_response,
    'SeasonalShoppingScore': seasonal_score,
    'TotalLifetimeValue': lifetime_value,
    'AvgOrderValue': avg_order_value,
    'RFM_Score': rfm_score,
    'SpendPerVisit': spend_per_visit,
    'TenurePerAge': tenure_per_age,
    'SpendPerTenure': spend_per_tenure,
    'SatisfactionPerTicket': satisfaction_per_ticket,
    'EngagementScore': engagement_score,
    'HighValueCustomer': high_value,
    'HighRiskFlag': high_risk,
    'CustomerSegment': customer_segment,
    'Gender_Encoded': gender_map[gender],
    'Category_Encoded': category_map[category]
}

input_df = pd.DataFrame([input_dict])

input_df = input_df[feature_columns]

if st.button("🔍 Predict Churn"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠️ This customer is likely to churn. (Probability: {proba:.2f})")
    else:
        st.success(f"✅ This customer is likely to stay. (Probability of churn: {proba:.2f})")
