import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ======================
# LOAD DATA
# ======================
url = "https://raw.githubusercontent.com/PrismaDana94/Onky-Pradana-Final-Project-Data-Science/main/Dana_Final_Project_Credit_Card_Fraud_detection.csv"
df_profit = pd.read_csv(url)

# Pastikan ada kolom profit per transaksi
# Kalau belum ada, kita buat manual (true positive = +100, false positive = -10)
if "profit" not in df_profit.columns:
    df_profit["profit"] = df_profit["y_true"] * 100 + (1 - df_profit["y_true"]) * -10
    df_profit["cum_profit"] = df_profit["profit"].cumsum()

# ======================
# APP TITLE
# ======================
st.title("💳 Fraud Detection – Profit Curve & Segmentation Dashboard")
st.markdown("""
Dashboard ini menampilkan hasil analisis model **XGBoost** untuk deteksi fraud,
dengan fokus pada **threshold, profit, dan segmentasi risiko**.
""")

# ======================
# 1. DATA PREVIEW
# ======================
st.subheader("📌 Data Preview")
st.dataframe(df_profit.head())

# ======================
# 2. PROFIT CURVE
# ======================
st.subheader("📈 Profit Curve")

max_profit_idx = df_profit['cum_profit'].idxmax()
threshold_opt = df_profit.loc[max_profit_idx, 'y_prob']
max_profit = df_profit['cum_profit'].max()
population_opt = (df_profit.loc[:max_profit_idx].shape[0] / df_profit.shape[0]) * 100

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(df_profit['population_pct'], df_profit['cum_profit'], label="Profit Curve")
ax.axhline(max_profit, color="green", linestyle="--", label=f"Max Profit: £{max_profit:,.0f}")
ax.axvline(population_opt, color="red", linestyle="--", label=f"Optimal Pop: {population_opt:.2f}%")
ax.set_xlabel("Percentage of Population Targeted (%)")
ax.set_ylabel("Cumulative Profit (£)")
ax.set_title("Profit Curve with Optimal Threshold")
ax.legend()
st.pyplot(fig)

st.table({
    "Metric": ["Optimal Threshold", "Max Profit (£)", "Target Population (%)"],
    "Value": [f"{threshold_opt:.4f}", f"{max_profit:,.0f}", f"{population_opt:.2f}%"]
})

# ======================
# 3. SIDEBAR THRESHOLD
# ======================
st.sidebar.header("⚙️ Controls")
user_threshold = st.sidebar.slider(
    "Set Threshold", 
    0.0, 0.05, float(threshold_opt), 0.001
)

# Segmentasi berdasarkan threshold user
df_profit['risk_segment'] = pd.cut(
    df_profit['y_prob'],
    bins=[0, user_threshold, 1],
    labels=['Low Risk', 'High Risk']
)

# ======================
# 4. RISK SEGMENTATION
# ======================
st.subheader("🔎 Risk Segmentation")

segment_counts = df_profit['risk_segment'].value_counts().sort_index()
segment_percent = (segment_counts / segment_counts.sum() * 100).round(2)

fig2, ax2 = plt.subplots(figsize=(6,4))
bars = ax2.bar(segment_counts.index, segment_counts.values, color=['skyblue', 'red'])
for bar, count, pct in zip(bars, segment_counts.values, segment_percent.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
             f"{count} ({pct}%)", ha='center', va='bottom')
ax2.set_title("Number of Transactions per Risk Segment")
ax2.set_ylabel("Count")
ax2.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig2)

fig3, ax3 = plt.subplots()
ax3.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%',
        colors=['skyblue', 'red'], startangle=90)
ax3.set_title("Risk Segment Distribution")
st.pyplot(fig3)

# ======================
# 5. PROFIT PER SEGMENT
# ======================
st.subheader("💰 Profit per Risk Segment")

segment_profit = df_profit.groupby('risk_segment')['profit'].sum().reset_index()
segment_profit['profit_pct'] = (segment_profit['profit'] / segment_profit['profit'].sum() * 100).round(2)

fig4, ax4 = plt.subplots(figsize=(6,4))
bars = ax4.bar(segment_profit['risk_segment'], segment_profit['profit'], color=['skyblue', 'red'])
for bar, profit, pct in zip(bars, segment_profit['profit'], segment_profit['profit_pct']):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
             f"£{profit:,.0f} ({pct}%)", ha='center', va='bottom')
ax4.set_title("Total Profit per Segment")
ax4.set_ylabel("Profit (£)")
ax4.grid(axis='y')
st.pyplot(fig4)

fig5, ax5 = plt.subplots()

# Plot bar
segment_profit.plot(
    kind='bar', x='risk_segment', y='profit', ax=ax5,
    color=['skyblue' if p >= 0 else 'red' for p in segment_profit['profit']]
)

# Tambahkan angka di atas bar
for i, v in enumerate(segment_profit['profit']):
    ax5.text(
        i, v + (0.02 * segment_profit['profit'].max()),  # posisi teks
        f"{v:,.0f}", ha='center', va='bottom', fontsize=10
    )

ax5.set_title("Profit per Risk Segment")
ax5.set_ylabel("Profit")
st.pyplot(fig5)



# ======================
# 6. INSIGHTS
# ======================
st.subheader("📊 Insights & Kesimpulan")
st.success(f"""
✅ Dengan threshold optimal {threshold_opt:.4f} (≈ {population_opt:.2f}% populasi ditarget),
kita dapat profit maksimum sebesar £{max_profit:,.0f}.
""")

st.info("""
- **Low Risk** = transaksi relatif aman → bisa diproses otomatis.  
- **High Risk** = transaksi dengan probabilitas fraud tinggi → perlu pengecekan manual.  
- Threshold bisa disesuaikan di sidebar untuk eksplorasi trade-off antara coverage & profit.
""")
