import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="MoneyXAI Fraud Detection", page_icon="💸", layout="centered")

# ---------------- CUSTOM UI (BLACK + GOLD THEME) ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

/* Titles */
h1, h2, h3 {
    color: #FFD700 !important;
    text-align: center;
}

/* Inputs */
.stTextInput input, .stNumberInput input {
    background-color: #111827;
    color: white;
    border-radius: 10px;
    border: 1px solid #FFD700;
    padding: 10px;
}

/* Selectbox */
.stSelectbox div {
    background-color: #111827 !important;
    color: white !important;
}

/* Button */
.stButton button {
    background: linear-gradient(90deg, #FFD700, #facc15);
    color: blue;
    font-weight: bold;
    border-radius: 10px;
    padding: 12px;
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #facc15, #FFD700);
}

/* Card */
.block-container {
    padding: 2rem;
    border-radius: 20px;
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(10px);
}

/* Result Boxes */
.success-box {
    padding:20px;
    border-radius:15px;
    background:rgba(0,255,0,0.1);
    border:1px solid green;
}

.error-box {
    padding:20px;
    border-radius:15px;
    background:rgba(255,0,0,0.1);
    border:1px solid red;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<h1>💸 MoneyXAI</h1>
<p style='text-align:center; font-size:18px; color:#cbd5e1;'>
AI-powered Anti-Money Laundering Detection System
</p>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_models():
    if os.path.exists('model.pkl') and os.path.exists('transformer.pkl'):
        model = joblib.load('model.pkl')
        transformer = joblib.load('transformer.pkl')
        return model, transformer
    return None, None

@st.cache_data
def load_sample_data():
    if os.path.exists('SAML-D.csv'):
        # Load a sample for the dashboard to keep it fast
        return pd.read_csv('SAML-D.csv', nrows=10000)
    return None

model, transformer = load_models()
sample_df = load_sample_data()

if model is None:
    st.error("🚨 Model files not found! Run `python train_and_save.py` first.")
    st.stop()

# ---------------- NAVIGATION ----------------
tab1, tab2, tab3 = st.tabs(["🔍 Predictor", "📊 Analytics Dashboard", "📂 Batch Processing"])

# ---------------- PREDICTOR TAB ----------------
with tab1:
    st.markdown("### 🧾 Transaction Input Panel")
    
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)

        with col1:
            sender_account = st.number_input("Sender Account", value=1234567890)
            receiver_account = st.number_input("Receiver Account", value=9876543210)
            amount = st.number_input("Amount", min_value=0.0, value=1500.0)

            payment_currency = st.text_input("Payment Currency", value="UK pounds")
            received_currency = st.text_input("Received Currency", value="UK pounds")

        with col2:
            sender_bank_loc = st.text_input("Sender Bank Location", value="UK")
            receiver_bank_loc = st.text_input("Receiver Bank Location", value="UK")

            payment_types = ["Credit card", "Debit card", "Cheque", "Cross-border", "ACH", "Cash Withdrawal", "Cash Deposit"]
            payment_type = st.selectbox("Payment Type", payment_types)

        st.markdown("---")
        st.subheader("Date & Time")

        col3, col4, col5, col6 = st.columns(4)

        with col3:
            year = st.number_input("Year", value=2023)

        with col4:
            month = st.number_input("Month", value=8)

        with col5:
            day = st.number_input("Day", value=15)

        with col6:
            week = st.number_input("Week", value=33)

        submit = st.form_submit_button("🚀 Analyze Transaction")

    # ---------------- PREDICTION ----------------
    if submit:
        amount_log = np.log1p(amount)

        data = pd.DataFrame([{
            'Sender_account': sender_account,
            'Receiver_account': receiver_account,
            'Amount': amount_log,
            'Payment_currency': payment_currency,
            'Received_currency': received_currency,
            'Sender_bank_location': sender_bank_loc,
            'Receiver_bank_location': receiver_bank_loc,
            'Payment_type': payment_type,
            'Year': year,
            'Month': month,
            'Day': day,
            'Week': week
        }])

        try:
            X_tf = transformer.transform(data)
            prediction = model.predict(X_tf)[0]
            probability = model.predict_proba(X_tf)[0][1]

            st.markdown("## 🔍 Analysis Result")

            if prediction == 1:
                st.markdown(f"""
                <div class="error-box">
                    <h3>🚨 Suspicious Transaction Detected</h3>
                    <p>Confidence: <b>{probability:.2%}</b></p>
                    <p>⚠️ Immediate action required</p>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="success-box">
                    <h3>✅ Normal Transaction</h3>
                    <p>Risk Score: <b>{probability:.2%}</b></p>
                    <p>✔️ No suspicious activity</p>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ---------------- DASHBOARD TAB ----------------
with tab2:
    st.markdown("### 📊 Money Laundering Analytics")
    
    if sample_df is not None:
        # Metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Total Transactions", len(sample_df))
        with col_m2:
            fraud_rate = (sample_df['Is_laundering'].sum() / len(sample_df)) * 100
            st.metric("Detected Fraud Rate", f"{fraud_rate:.2f}%")
        with col_m3:
            st.metric("Avg Amount", f"${sample_df['Amount'].mean():,.2f}")

        # Charts
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("#### Transaction Types")
            fig_pie = px.pie(sample_df, names='Payment_type', hole=0.4, 
                            color_discrete_sequence=px.colors.sequential.YlOrBr)
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_c2:
            st.markdown("#### Amount Distribution")
            fig_hist = px.histogram(sample_df, x='Amount', color='Is_laundering', 
                                   marginal="box", barmode="overlay",
                                   color_discrete_map={0: "#0ea5e9", 1: "#ef4444"})
            fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig_hist, use_container_width=True)

        # Feature Importance
        st.markdown("#### 🤖 AI Model Logic (Feature Importance)")
        importances = model.feature_importances_
        # Get feature names from transformer
        try:
            # Get the feature names after transformation if possible, or use raw if simplified
            # For simplicity in this demo, we'll use a subset of common names
            feature_names = ['Account (Sender)', 'Account (Receiver)', 'Amount', 'Currency', 'Recv Currency', 'Sender Loc', 'Recv Loc', 'Type', 'Year', 'Month', 'Day', 'Week']
            # Crop/pad as needed
            importances = importances[:len(feature_names)]
            fig_imp = px.bar(x=importances, y=feature_names[:len(importances)], orientation='h',
                            color_discrete_sequence=['#FFD700'])
            fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white",
                                 xaxis_title="Importance Score", yaxis_title="Features")
            st.plotly_chart(fig_imp, use_container_width=True)
        except:
            st.info("Feature importance data currently being updated.")

    else:
        st.warning("⚠️ No data available for the dashboard. Please check `SAML-D.csv`.")

# ---------------- BATCH PROCESSING TAB ----------------
with tab3:
    st.markdown("### 📂 Batch Transaction Analysis")
    st.write("Upload a CSV file with multiple transactions to analyze them in bulk.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:")
            st.dataframe(batch_df.head())
            
            if st.button("🚀 Analyze Batch"):
                # Preprocess batch
                # To keep it simple, we assume the CSV has the same columns
                # We need to apply log1p to Amount
                temp_df = batch_df.copy()
                if 'Amount' in temp_df.columns:
                    temp_df['Amount'] = np.log1p(temp_df['Amount'])
                
                # Apply transformer
                X_batch_tf = transformer.transform(temp_df)
                predictions = model.predict(X_batch_tf)
                probs = model.predict_proba(X_batch_tf)[:, 1]
                
                batch_df['Suspicious'] = predictions
                batch_df['Risk_Score'] = probs
                
                st.markdown("#### 🔍 Results")
                fraud_count = batch_df['Suspicious'].sum()
                st.warning(f"🚨 Detected {fraud_count} suspicious transactions out of {len(batch_df)}")
                
                st.dataframe(batch_df.sort_values(by='Risk_Score', ascending=False))
                
                # Download button
                csv_out = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results", data=csv_out, file_name="analysis_results.csv", mime="text/csv")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")