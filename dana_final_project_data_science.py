import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

url = "https://raw.githubusercontent.com/PrismaDana94/Onky-Pradana-Final-Project-Data-Science/main/Dana_Final_Project_Credit_Card_Fraud_detection.csv"

df_profit = pd.read_csv(url)

st.title("Fraud Detection – Profit Curve & Segmentation Dashboard")
st.markdown("""
Dashboard ini menampilkan hasil analisis model **XGBoost** untuk mendeteksi fraud, 
dengan fokus pada **threshold, profit, dan segmentasi risiko**.
""")


# ======================
# 1. DATA PREVIEW
# ======================
st.subheader("Data Preview")
st.dataframe(df_profit.head())

# ======================
# 2. Profit Curve & Threshold Optimal
# ======================
st.subheader("Profit Curve")

# Cari threshold optimal
max_profit_idx = df_profit['cum_profit'].idxmax()
threshold = df_profit.loc[max_profit_idx, 'y_prob']
max_profit = df_profit['cum_profit'].max()
population_optimal = (df_profit.loc[:max_profit_idx].shape[0] / df_profit.shape[0]) * 100

# Plot Profit Curve
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.plot(df_profit['y_prob'], df_profit['cum_profit'], label="Profit Curve")
ax2.axvline(threshold, color='red', linestyle='--', label=f"Threshold: {threshold:.4f}")
ax2.set_xlabel("Probability Threshold")
ax2.set_ylabel("Cumulative Profit")
ax2.set_title("Profit Curve with Optimal Threshold")
ax2.legend()
st.pyplot(fig2)

# Summary metrics
st.subheader("Summary Metrics")
st.table({
    "Metric": ["Optimal Threshold", "Max Profit (£)", "Target Population (%)"],
    "Value": [f"{threshold:.4f}", f"{max_profit:,.0f}", f"{population_optimal:.2f}%"]
})

# ======================
# 3. Risk Segmentation
# ======================
st.subheader("Risk Segmentation (2 Segments)")

df_profit['risk_segment'] = pd.cut(
    df_profit['y_prob'],
    bins=[0, threshold, 1],
    labels=['Low Risk', 'High Risk']
)

segment_counts = df_profit['risk_segment'].value_counts().sort_index()
segment_percent = (segment_counts / segment_counts.sum() * 100).round(2)

# Bar chart
fig3, ax3 = plt.subplots(figsize=(6,4))
bars = ax3.bar(segment_counts.index, segment_counts.values, color=['skyblue', 'red'])
for bar, count, pct in zip(bars, segment_counts.values, segment_percent.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()+200,
             f'{count} ({pct}%)', ha='center', va='bottom', fontsize=10)
ax3.set_title("Number of Transactions per Risk Segment")
ax3.set_ylabel("Number of Transactions")
ax3.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig3)

# Pie chart
fig4, ax4 = plt.subplots()
ax4.pie(segment_counts, labels=segment_counts.index,
        autopct='%1.1f%%', colors=['skyblue', 'red'], startangle=90)
ax4.set_title("Risk Segment Distribution")
st.pyplot(fig4)

# ======================
# 4. Threshold Interaktif
# ======================
st.subheader("Explore Different Thresholds")

user_threshold = st.slider("Set Threshold", 0.0, 0.05, float(threshold), 0.001)

df_profit['user_segment'] = pd.cut(
    df_profit['y_prob'],
    bins=[0, user_threshold, 1],
    labels=['Low Risk', 'High Risk']
)

user_counts = df_profit['user_segment'].value_counts()
user_percent = (user_counts / user_counts.sum() * 100).round(2)

fig5, ax5 = plt.subplots()
bars = ax5.bar(user_counts.index, user_counts.values, color=['skyblue', 'red'])
for bar, count, pct in zip(bars, user_counts.values, user_percent.values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height()+200,
             f'{count} ({pct}%)', ha='center', va='bottom', fontsize=10)
ax5.set_title(f"Transactions by Segment (Threshold = {user_threshold:.4f})")
st.pyplot(fig5)

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


# ======================
# 5. Insights
# ======================
st.subheader("Insights & Kesimpulan")

st.success(f"""
✅ Dengan threshold optimal {threshold:.4f} (≈ {population_optimal:.2f}% populasi ditarget), 
kita dapat profit maksimum sebesar £{max_profit:,.0f}.
""")

st.info("""
- **Low Risk** = transaksi yang relatif aman.  
- **High Risk** = transaksi dengan probabilitas fraud tinggi, lebih fokus dicek manual.  
- Threshold bisa disesuaikan untuk eksplorasi trade-off antara coverage & profit.
""")











