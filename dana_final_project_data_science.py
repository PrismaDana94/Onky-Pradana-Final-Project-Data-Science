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











