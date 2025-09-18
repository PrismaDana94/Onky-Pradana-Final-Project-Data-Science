# Final Project – Credit Card Fraud Detection

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen)](https://onky-pradana-final-project-data-science3.streamlit.app/)

##  Project Overview

Proyek ini berfokus pada pembuatan **dashboard interaktif dengan Streamlit** untuk menganalisis potensi fraud pada transaksi kartu kredit.
Dashboard ini menampilkan **Profit Curve Analysis** untuk menentukan threshold optimal yang memaksimalkan keuntungan bisnis, serta **Risk Segmentation** untuk memisahkan transaksi aman dan berisiko tinggi.

---

## Dataset
Dataset ini merupakan hasil simulasi transaksi kartu kredit dengan label probabilitas fraud.

Dataset yang digunakan diambil langsung dari repo ini:  
[`Final_Project_Credit_Card_Fraud_detection.csv`](./Final_Project_Credit_Card_Fraud_detection.csv)

Beberapa kolom penting:

- y_prob → probabilitas fraud yang sudah diprediksi dari model sebelumnya
- cum_profit → perhitungan kumulatif profit berdasarkan threshold

---

## 🚀 Dashboard Workflow
1. **Data Loading** → membaca dataset hasil prediksi probabilitas fraud.  
2. **Profit Curve Analysis** → menampilkan kurva profit dan menentukan threshold optimal.  
3. **Risk Segmentation** → membagi transaksi ke dalam kategori **Low Risk** & **High Risk**.  
4. **Threshold Explorer** → slider interaktif untuk menganalisis trade-off antara profit dan coverage.  
5. **Dashboard Output** → visualisasi interaktif (tabel, bar chart, pie chart) serta ringkasan insights.
   
---

## 📈 Dashboard Features
- **Data Preview** – melihat dataset hasil prediksi probabilitas fraud  
- **Profit Curve** – menampilkan threshold optimal & profit maksimum  
- **Risk Segmentation** – visualisasi distribusi transaksi per kategori risiko  
- **Threshold Explorer** – slider interaktif untuk menganalisis trade-off  
- **Profit per Risk Segment** – membandingkan total profit tiap segmen risiko  
- **Insights** – ringkasan hasil analisis & rekomendasi bisnis
  
🔗 **Live App** → [Streamlit Dashboard](https://onky-pradana-final-project-data-science3.streamlit.app/)


---
## 🛠️ Tech Stack
- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost)  
- Streamlit – dashboard interaktif  
- GitHub – version control & deployment  

---

## 📌 Key Insights
Berdasarkan analisis pada dataset yang tersedia:  
- Threshold optimal berada di sekitar **0.0093** → menargetkan ~15.9% populasi.  
- Profit maksimum yang bisa dicapai adalah sekitar **£124.780**.  
- **Low Risk** → transaksi aman, dapat diproses otomatis.  
- **High Risk** → transaksi dengan risiko tinggi, disarankan untuk pengecekan manual.
Hasil ini hanya berlaku pada dataset yang tersedia, bukan hasil training model dari repo ini.
---

## 📂 Main Files
- `Final_Project_Credit_Card_Fraud_detection.csv` – dataset hasil prediksi  
- `final_project_data_science.py` – Streamlit dashboard  

---

## 📬 Author
👤 **Onky Pradana**  
- 📧 Email: [freddycull27@gmail.com]  
- 💼 LinkedIn:  [my_linkedin](https://www.linkedin.com/in/prisma-dana/)  
- 🐙 GitHub: [PrismaDana94](https://github.com/PrismaDana94)

---

✨ Feel free to fork & explore this repo!  
Jika ada pertanyaan atau saran, silakan open issue di repo ini.

