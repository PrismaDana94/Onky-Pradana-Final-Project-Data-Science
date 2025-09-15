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

st.subheader("Profit per Risk Segment")

# Hitung total profit per segmen (pakai sum dari cum_profit)
segment_profit = df_profit.groupby('risk_segment')['cum_profit'].sum().reset_index()
segment_profit = segment_profit.rename(columns={'cum_profit': 'total_profit'})

# Hitung persentase profit
segment_profit['profit_pct'] = (segment_profit['total_profit'] / segment_profit['total_profit'].sum() * 100).round(2)

# --- Bar Chart Profit ---
fig3, ax3 = plt.subplots(figsize=(6,4))
bars = ax3.bar(segment_profit['risk_segment'], segment_profit['total_profit'], color=['skyblue', 'red'])

# Tambahkan label profit di atas bar
for bar, profit, pct in zip(bars, segment_profit['total_profit'], segment_profit['profit_pct']):
    ax3.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + (bar.get_height()*0.02),
        f"£{profit:,.0f} ({pct}%)",
        ha='center', va='bottom'
    )

ax3.set_title("Total Profit per Risk Segment")
ax3.set_xlabel("Risk Segment")
ax3.set_ylabel("Total Profit (£)")
ax3.grid(axis='y')
st.pyplot(fig3)

# --- Pie Chart Profit ---
fig4, ax4 = plt.subplots(figsize=(4,4))
ax4.pie(
    segment_profit['total_profit'],
    labels=segment_profit['risk_segment'],
    autopct='%1.1f%%',
    colors=['skyblue', 'red'],
    startangle=90
)
ax4.set_title("Profit Distribution per Segment")
st.pyplot(fig4)






