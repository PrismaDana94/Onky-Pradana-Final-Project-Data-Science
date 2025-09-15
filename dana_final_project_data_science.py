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

max_profit_idx = df_profit['cum_profit'].idxmax()
threshold = df_profit.loc[max_profit_idx, 'y_prob']
st.write("Threshold Optimal dari Profit Curve:", threshold)

st.write("Jumlah baris data:", len(df_profit))
st.write(df_profit['y_prob'].describe())
segment_counts = df_profit['risk_segment'].value_counts().sort_index()
st.write("Segment Counts:", segment_counts)
segment_counts = df_profit['risk_segment'].value_counts().sort_index()
segment_percent = (segment_counts / segment_counts.sum() * 100).round(2)

st.write("Segment Counts:", segment_counts)
st.write("Segment Percentages:", segment_percent)
# --- Segmentasi Risiko ---
threshold = 0.0093
df_profit['risk_segment'] = pd.cut(
    df_profit['y_prob'],
    bins=[0, threshold, 1],
    labels=['Low Risk', 'High Risk']
)

# Hitung jumlah dan persentase
segment_counts = df_profit['risk_segment'].value_counts().sort_index()
segment_percent = (segment_counts / segment_counts.sum() * 100).round(2)

st.subheader("Segmentasi Risiko Fraud")

# --- Bar Chart ---
fig, ax = plt.subplots(figsize=(6,4))
bars = ax.bar(segment_counts.index, segment_counts.values, color=['skyblue', 'red'])

# Tambahkan label jumlah & persentase di atas bar
for bar, count, pct in zip(bars, segment_counts.values, segment_percent.values):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 100,
        f"{count} ({pct}%)",
        ha='center', va='bottom'
    )

ax.set_title("Number of Transactions per Risk Segment")
ax.set_xlabel("Risk Segment")
ax.set_ylabel("Number of Transactions")
ax.grid(axis='y')
st.pyplot(fig)

# --- Pie Chart ---
fig2, ax2 = plt.subplots(figsize=(4,4))
ax2.pie(
    segment_percent,
    labels=segment_percent.index,
    autopct='%1.1f%%',
    colors=['skyblue', 'red'],
    startangle=90
)
ax2.set_title("Percentage of Transactions per Segment")
st.pyplot(fig2)





