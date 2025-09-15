# app.py
import streamlit as st
import pandas as pd

# Judul aplikasi
st.title("Dana Final Project - Credit Card Fraud Detection")

# URL file CSV di GitHub (raw link, bukan link biasa!)
url = "https://raw.githubusercontent.com/PrismaDana94/Onky-Pradana-Final-Project-Data-Science/main/Dana_Final_Project_Credit_Card_Fraud_detection.csv"

# Baca CSV
df = pd.read_csv(url)

# Tampilkan 5 baris pertama
st.subheader("Data Preview")
st.write(df.head())
