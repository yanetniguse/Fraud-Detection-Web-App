# fraud_app_final.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier  # example, use your model

# Load model and feature list from files in your repo
model = joblib.load("fraud_model.pkl")
features = joblib.load("model_features.pkl")

# save with joblib
joblib.dump(model, "fraud_model.pkl")
joblib.dump(features, "model_features.pkl")

# -------------------------------
# Key features for manual input & CSV output
# -------------------------------
key_inputs = [
    "AVAIL_CRDT",
    "AMOUNT",
    "CREDIT_LIMIT",
    "CARD_NOT_PRESENT",
    "TIME_SPENT"
]

st.set_page_config(page_title="💳 Fraud Detection", layout="wide")
st.title("💳 Fraud Detection Web App")

st.write("Enter transaction details manually or upload a CSV/JSON file.")
st.markdown("""
**Units for input:**  
- Available Credit, Transaction Amount, Credit Limit → USD 💵  
- Time Spent on Transaction → seconds ⏱  

**Expected CSV/JSON Columns (case-sensitive):**  
`AVAIL_CRDT, AMOUNT, CREDIT_LIMIT, CARD_NOT_PRESENT, TIME_SPENT`
""")

# -------------------------------
# 1️⃣ CSV / JSON Upload Section
# -------------------------------
uploaded_file = st.file_uploader("Upload transaction CSV/JSON", type=["csv", "json"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        input_df = pd.read_csv(uploaded_file)
    else:
        input_df = pd.read_json(uploaded_file)

    # Fill missing features with 0 (backend features)
    for feature in features:
        if feature not in input_df.columns:
            input_df[feature] = 0

    # Ensure correct column order for model
    input_df = input_df[features]

    st.write("✅ Uploaded file preview (key columns only):")
    st.dataframe(input_df[key_inputs].head())

    if st.button("Predict from Uploaded File"):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[:, 1]

        # Prepare clean CSV: only key features + prediction results
        clean_df = input_df[key_inputs].copy()
        clean_df["Prediction"] = ["Fraud" if x == 1 else "Normal" for x in prediction]
        clean_df["Fraud_Probability"] = prediction_proba
        clean_df["Risk_Level"] = pd.cut(
            prediction_proba,
            bins=[-0.01, 0.5, 0.75, 1.0],
            labels=["Low", "Medium", "High"]
        )

        st.subheader("📊 Predictions")
        st.dataframe(clean_df.style.applymap(
            lambda x: 'background-color: #ffcccc' if x=="Fraud" else 'background-color: #ccffcc',
            subset=["Prediction"]
        ))

        # CSV download
        csv = clean_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

# -------------------------------
# 2️⃣ Manual Input Section
# -------------------------------
st.subheader("📝 Manual Transaction Input")
st.write("Fill in key transaction details below. Units are shown for clarity.")

with st.form("manual_input_form"):
    user_input = {}
    user_input["AVAIL_CRDT"] = st.number_input(
        "Available Credit (USD) 💵", value=0.0, step=100.0,
        help="Amount of available credit on the card in USD"
    )
    user_input["AMOUNT"] = st.number_input(
        "Transaction Amount (USD) 💵", value=0.0, step=50.0,
        help="The amount of the transaction in USD"
    )
    user_input["CREDIT_LIMIT"] = st.number_input(
        "Credit Limit (USD) 💳", value=0.0, step=100.0,
        help="Maximum credit limit of the card in USD"
    )
    user_input["CARD_NOT_PRESENT"] = st.radio(
        "Card Present? 🏦",
        options=[0, 1],
        format_func=lambda x: "Card Present" if x == 0 else "Card Not Present",
        help="Select whether the physical card was present during the transaction"
    )
    user_input["TIME_SPENT"] = st.number_input(
        "Time Spent on Transaction (seconds) ⏱", value=0.0, step=5.0,
        help="Time taken by the user to complete the transaction in seconds"
    )

    submitted = st.form_submit_button("Predict from Manual Input")

if submitted:
    manual_df = pd.DataFrame([user_input])
    # Fill missing backend features with 0
    for feature in features:
        if feature not in manual_df.columns:
            manual_df[feature] = 0
    manual_df = manual_df[features]

    pred = model.predict(manual_df)[0]
    pred_proba = model.predict_proba(manual_df)[0][1]

    # Determine risk level
    if pred_proba > 0.75:
        risk = "High ⚠️"
        color = "#ff4d4d"
    elif pred_proba > 0.5:
        risk = "Medium ⚠️"
        color = "#ffcc66"
    else:
        risk = "Low ✅"
        color = "#b3ffb3"

    st.subheader("📌 Prediction Result")
    st.markdown(
        f"<div style='background-color:{color};padding:10px;border-radius:5px;'>"
        f"**Prediction:** {'Fraud' if pred==1 else 'Normal'}  \n"
        f"**Fraud Probability:** {pred_proba:.2%}  \n"
        f"**Risk Level:** {risk}"
        f"</div>",
        unsafe_allow_html=True
    )

    # Explain prediction with simple reasoning
    st.subheader("🧐 Explanation (simple)")
    reasons = []
    if user_input["CARD_NOT_PRESENT"] == 1:
        reasons.append("Card not present")
    if user_input["AMOUNT"] > (0.5 * user_input["CREDIT_LIMIT"]):
        reasons.append("High transaction relative to credit limit")
    if user_input["TIME_SPENT"] < 30:
        reasons.append("Transaction completed very quickly")

    if reasons:
        st.write("⚠️ Factors contributing to fraud prediction:")
        for r in reasons:
            st.write(f"- {r}")
    else:
        st.write("✅ No obvious red flags detected in key inputs.")

    # Optional: show local currency (Ksh) conversion
    usd_to_ksh = 150  # example rate, adjust as needed
    st.write(f"💱 Equivalent Transaction Amount: Ksh {user_input['AMOUNT']*usd_to_ksh:,.2f}")
    st.write(f"💱 Available Credit: Ksh {user_input['AVAIL_CRDT']*usd_to_ksh:,.2f}")
    st.write(f"💱 Credit Limit: Ksh {user_input['CREDIT_LIMIT']*usd_to_ksh:,.2f}")



