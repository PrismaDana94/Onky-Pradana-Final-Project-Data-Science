import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Dana Final Project - Credit Card Fraud Detection")

# Load data dari GitHub
url = "https://raw.githubusercontent.com/username/repo/main/df_profit.csv"  # ganti sesuai repo kamu
df_profit = pd.read_csv(url)

st.subheader("Data Preview ↩️")
st.dataframe(df_profit.head())

# --- Deskripsi Data ---
st.subheader("Deskripsi Data")
st.write(f"Jumlah transaksi: {len(df_profit):,}")
st.write(f"Jumlah fraud: {df_profit['y_true'].sum():,}")
st.write(f"Jumlah non-fraud: {len(df_profit) - df_profit['y_true'].sum():,}")

# --- Pie Chart Fraud vs Non-Fraud ---
fraud_counts = df_profit['y_true'].value_counts()

fig, ax = plt.subplots()
ax.pie(fraud_counts, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90, colors=["#4CAF50", "#FF5722"])
ax.axis('equal')  # biar pie chart bulat

st.pyplot(fig)

