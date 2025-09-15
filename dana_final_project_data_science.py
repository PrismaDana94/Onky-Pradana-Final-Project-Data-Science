import streamlit as st
import pandas as pd

# link raw github
url = "https://raw.githubusercontent.com/PrismaDana94/Onky-Pradana-Final-Project-Data-Science/main/Dana_Final_Project_Credit_Card_Fraud_detection.csv"

# load data
df_profit = pd.read_csv(url)

# tampilkan preview
st.title("Dana Final Project - Credit Card Fraud Detection")
st.write("Data Preview", df_profit.head())

