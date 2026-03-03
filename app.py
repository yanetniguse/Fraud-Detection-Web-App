# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="💳 AI Fraud Detection Dashboard", layout="wide")

st.title("💳 AI-Powered Fraud Detection Dashboard")
st.markdown("Predict fraud risk, analyze transactions, and explore model insights.")

# -------------------------------
# Load Model & Features Safely
# -------------------------------
try:
    model = joblib.load("fraud_model.pkl")
    features = joblib.load("model_features.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Key visible inputs
key_inputs = [
    "AVAIL_CRDT",
    "AMOUNT",
    "CREDIT_LIMIT",
    "CARD_NOT_PRESENT",
    "TIME_SPENT"
]

# -------------------------------
# Create Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs([
    "📝 Manual Prediction",
    "📂 Upload & Analyze",
    "📊 Model Insights"
])

# =========================================================
# 📝 TAB 1 — MANUAL PREDICTION
# =========================================================
with tab1:

    st.subheader("Enter Transaction Details")

    with st.form("manual_form"):
        user_input = {}

        col1, col2 = st.columns(2)

        with col1:
            user_input["AVAIL_CRDT"] = st.number_input("Available Credit (USD)", value=0.0)
            user_input["AMOUNT"] = st.number_input("Transaction Amount (USD)", value=0.0)
            user_input["CREDIT_LIMIT"] = st.number_input("Credit Limit (USD)", value=0.0)

        with col2:
            user_input["CARD_NOT_PRESENT"] = st.radio(
                "Card Present?",
                options=[0, 1],
                format_func=lambda x: "Card Present" if x == 0 else "Card Not Present"
            )
            user_input["TIME_SPENT"] = st.number_input("Time Spent (seconds)", value=0.0)

        submitted = st.form_submit_button("🔍 Predict Fraud Risk")

    if submitted:

        manual_df = pd.DataFrame([user_input])

        for feature in features:
            if feature not in manual_df.columns:
                manual_df[feature] = 0

        manual_df = manual_df[features]

        pred = model.predict(manual_df)[0]
        prob = model.predict_proba(manual_df)[0][1]

        # Risk Level
        if prob < 0.4:
            risk_label = "🟢 Low Risk"
            color = "#b3ffb3"
        elif prob < 0.75:
            risk_label = "🟡 Medium Risk"
            color = "#ffe680"
        else:
            risk_label = "🔴 High Risk"
            color = "#ff9999"

        st.subheader("📌 Prediction Result")
        st.markdown(
            f"""
            <div style='background-color:{color};padding:15px;border-radius:8px;'>
            <h4>Prediction: {'Fraud' if pred==1 else 'Normal'}</h4>
            <h4>Fraud Probability: {prob:.2%}</h4>
            <h4>Risk Level: {risk_label}</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Risk Meter
        st.subheader("📊 Risk Meter")
        st.progress(float(prob))

        # Simple Explanation
        st.subheader("🧐 Key Risk Indicators")
        reasons = []

        if user_input["CARD_NOT_PRESENT"] == 1:
            reasons.append("Card not present during transaction")
        if user_input["CREDIT_LIMIT"] > 0 and user_input["AMOUNT"] > 0.5 * user_input["CREDIT_LIMIT"]:
            reasons.append("High transaction relative to credit limit")
        if user_input["TIME_SPENT"] < 30:
            reasons.append("Transaction completed very quickly")

        if reasons:
            for r in reasons:
                st.write(f"- {r}")
        else:
            st.write("No strong red flags detected in key features.")
        st.session_state["last_manual_input"] = user_input.copy()
# =========================================================
# 📂 TAB 2 — UPLOAD & ANALYZE
# =========================================================
with tab2:

    st.subheader("Upload CSV or JSON File")

    uploaded_file = st.file_uploader("Upload transaction file", type=["csv", "json"])

    if uploaded_file:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_json(uploaded_file)

        for feature in features:
            if feature not in df.columns:
                df[feature] = 0

        df_model = df[features]

        st.write("Preview of Uploaded Data:")
        st.dataframe(df[key_inputs].head())

        if st.button("📊 Run Fraud Analysis"):

            preds = model.predict(df_model)
            probs = model.predict_proba(df_model)[:, 1]

            results = df[key_inputs].copy()
            results["Prediction"] = ["Fraud" if x == 1 else "Normal" for x in preds]
            results["Fraud_Probability"] = probs
            results["Risk_Level"] = pd.cut(
                probs,
                bins=[-0.01, 0.4, 0.75, 1.0],
                labels=["Low", "Medium", "High"]
            )

            st.subheader("📋 Prediction Results")
            st.dataframe(results)

            # Analytics
            st.subheader("📈 Dataset Analytics")

            total = len(results)
            fraud_count = sum(preds)
            fraud_rate = fraud_count / total if total > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", total)
            col2.metric("Fraud Transactions", fraud_count)
            col3.metric("Fraud Rate", f"{fraud_rate:.2%}")

            st.subheader("Risk Distribution")
            st.bar_chart(results["Risk_Level"].value_counts())

            # Download
            csv = results.to_csv(index=False)
            st.download_button(
                label="⬇ Download Results CSV",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

# =========================================================
# 📊 TAB 3 — MODEL INSIGHTS
# =========================================================
# -------------------------------
# TAB 3 — MODEL INSIGHTS / Transaction-Level Explanation
# -------------------------------
with tab3:
    st.subheader("🎯 Transaction-Level Explanation")
    st.write(
        "See how each feature influences the prediction for a single transaction. "
        "If you predicted a transaction in Tab 1 (Manual Input), you can use that input here."
    )

    # Check if there is a last manual input
    if "last_manual_input" in st.session_state:
        st.info("Using last transaction from Manual Input.")

        explain_df = pd.DataFrame([st.session_state["last_manual_input"]])

        # Show the transaction values in a clean table
        st.table(explain_df.T.rename(columns={0: "Value"}))

        # Button to generate explanation
        if st.button("🔍 Explain This Transaction"):
            # Convert to model-ready format
            model_input = explain_df.copy()
            for feature in features:
                if feature not in model_input.columns:
                    model_input[feature] = 0
            model_input = model_input[features]

            # Prediction & probability (again for display)
            pred = model.predict(model_input)[0]
            pred_proba = model.predict_proba(model_input)[0][1]

            # Display prediction summary
            st.markdown(
                f"**Prediction:** {'Fraud' if pred==1 else 'Normal'}  \n"
                f"**Fraud Probability:** {pred_proba:.2%}"
            )

            # -------------------------------
            # Feature influence / simple explanation
            # -------------------------------
            # Create a scaled importance (simple heuristic)
            importance = {}
            row = explain_df.iloc[0]
            # Example logic for influence scaling
            importance["Transaction Amount"] = row["AMOUNT"] / max(row["CREDIT_LIMIT"], 1)
            importance["Available Credit"] = row["AVAIL_CRDT"] / max(row["CREDIT_LIMIT"], 1)
            importance["Card Not Present"] = row["CARD_NOT_PRESENT"]
            importance["Time Spent"] = max(0, (120 - row["TIME_SPENT"])) / 120  # fast txn = higher influence

            # Convert to DataFrame for plotting
            imp_df = pd.DataFrame(list(importance.items()), columns=["Feature", "Influence"])
            imp_df["Influence"] = imp_df["Influence"].apply(lambda x: min(max(x, 0), 1))  # clip 0-1

            # Plot feature influence
            import altair as alt
            chart = (
                alt.Chart(imp_df)
                .mark_bar()
                .encode(
                    x=alt.X("Influence", title="Influence on Fraud Prediction (0-1)"),
                    y=alt.Y("Feature", sort='-x', title="Feature"),
                    color=alt.Color("Influence", scale=alt.Scale(scheme="reds"))
                )
                .properties(height=250, width=500, title="Feature Influence for This Transaction")
            )
            st.altair_chart(chart)

    else:
        st.warning(
            "No transaction found from Tab 1. Please input a transaction in Manual Input first, "
            "or enter a transaction manually below to see the explanation."
        )
        with st.expander("Manual Transaction Entry for Explanation"):
            manual_input = {}
            manual_input["AVAIL_CRDT"] = st.number_input("Available Credit (USD) 💵", value=0.0, step=100.0)
            manual_input["AMOUNT"] = st.number_input("Transaction Amount (USD) 💵", value=0.0, step=50.0)
            manual_input["CREDIT_LIMIT"] = st.number_input("Credit Limit (USD) 💳", value=0.0, step=100.0)
            manual_input["CARD_NOT_PRESENT"] = st.radio(
                "Card Present?", options=[0,1], format_func=lambda x: "Card Present" if x==0 else "Card Not Present"
            )
            manual_input["TIME_SPENT"] = st.number_input("Time Spent on Transaction (seconds) ⏱", value=0.0, step=5.0)
            
            if st.button("🔍 Explain Entered Transaction"):
                # store temporarily in session to reuse
                st.session_state["last_manual_input"] = manual_input
                st.experimental_rerun()  # refresh tab to show explanation
