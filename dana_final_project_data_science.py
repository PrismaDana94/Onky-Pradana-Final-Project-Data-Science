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




