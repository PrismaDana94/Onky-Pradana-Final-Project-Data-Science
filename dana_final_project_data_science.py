import pandas as pd

url = "https://raw.githubusercontent.com/PrismaDana94/Onky-Pradana-Final-Project-Data-Science/main/Dana_Final_Project_Credit_Card_Fraud_detection.csv"
df_profit = pd.read_csv(url)

st.write("Data Preview", df_profit.head())
