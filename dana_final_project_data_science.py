import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

url = "https://raw.githubusercontent.com/PrismaDana94/Onky-Pradana-Final-Project-Data-Science/main/Dana_Final_Project_Credit_Card_Fraud_detection.csv"

df_profit = pd.read_csv(url)

st.title("Dana Final Project - Credit Card Fraud Detection")
st.write("Preview Data:")
st.dataframe(df_profit.head())

import matplotlib.pyplot as plt

# Plot Profit Curve
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df_profit['population_pct'], df_profit['cum_profit'], color="green", label="Profit Curve")
ax.set_xlabel("Percentage of Population Targeted (%)")
ax.set_ylabel("Cumulative Profit (£)")
ax.set_title("Profit Curve for Fraud Detection")
ax.grid(True)
ax.legend()

# Tampilkan ke Streamlit
st.pyplot(fig)

# Hitung profit per segmen (misalnya sudah ada kolom risk_segment)
if "risk_segment" in df_profit.columns:
    segment_profit = df_profit.groupby('risk_segment')['cum_profit'].sum().reset_index()

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.pie(segment_profit['cum_profit'], labels=segment_profit['risk_segment'], autopct='%1.1f%%', startangle=90)
    ax2.set_title("Profit Share per Risk Segment")

    st.pyplot(fig2)



