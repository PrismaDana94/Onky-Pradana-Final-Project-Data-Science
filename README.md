# Final Project – Credit Card Fraud Detection

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen)](https://onky-pradana-final-project-data-science.streamlit.app/)

##  Project Overview
Proyek ini bertujuan untuk membangun **model machine learning** dalam mendeteksi transaksi kartu kredit yang berpotensi fraud.  
Selain modeling, dibuat juga **Profit Curve Analysis** untuk menentukan threshold optimal yang memaksimalkan keuntungan bisnis, serta segmentasi risiko untuk mendukung pengambilan keputusan.

---

## Dataset
- **Sumber Data:** Simulasi dataset transaksi kartu kredit.  
- **Jumlah Data:** ± 20.000 transaksi.  
- **Kolom utama:**  
  - `y_prob` → probabilitas fraud dari model XGBoost  
  - `cum_profit` → perhitungan kumulatif profit berdasarkan threshold  

Dataset yang digunakan diambil langsung dari repo ini:  
[`Dana_Final_Project_Credit_Card_Fraud_detection.csv`](./Dana_Final_Project_Credit_Card_Fraud_detection.csv)

---

## 🚀 Metodologi
1. **Exploratory Data Analysis (EDA)** → memahami distribusi transaksi & fraud.  
2. **Modeling dengan XGBoost** → prediksi probabilitas fraud.  
3. **Profit Curve Analysis** → menentukan threshold optimal.  
4. **Risk Segmentation** → membagi transaksi ke dalam **Low Risk** & **High Risk**.  
5. **Dashboard Streamlit** → interaktif untuk eksplorasi threshold & dampaknya terhadap profit.

---

## 📈 Dashboard Features
✅ **Data Preview** – menampilkan dataset hasil prediksi  
✅ **Profit Curve** – dengan threshold optimal & profit maksimum  
✅ **Risk Segmentation** – bar chart & pie chart distribusi transaksi  
✅ **Threshold Explorer** – slider interaktif untuk analisis trade-off  
✅ **Profit per Risk Segment** – membandingkan total profit per kategori risiko  
✅ **Insights** – ringkasan hasil analisis & rekomendasi bisnis  

🔗 **Live App** → [Streamlit Dashboard](https://onky-pradana-final-project-data-science.streamlit.app/)

---

## 🛠️ Tech Stack
- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost)  
- **Streamlit** – dashboard interaktif  
- **GitHub** – version control & deployment  

---

## 📌 Key Insights
- Threshold optimal berada di sekitar **0.0093** → menargetkan **~15.9% populasi**.  
- Profit maksimum yang bisa dicapai adalah sekitar **£X,XXX,XXX** (lihat dashboard).  
- **Low Risk** → transaksi aman, bisa diproses otomatis.  
- **High Risk** → transaksi dengan risiko tinggi, disarankan untuk pengecekan manual.  

---

## 📂 Project Structure
├── Dana_Final_Project_Credit_Card_Fraud_detection.csv # Dataset
├── dana_final_project_data_science.py # Streamlit app
├── requirements.txt # Dependencies
└── README.md # Project documentation

---

## 📬 Author
👤 **Onky Pradana**  
- 📧 Email: [freddycull27@gmail.com]  
- 💼 LinkedIn:[your_linkedin](https://www.linkedin.com/in/prisma-dana/)  
- 🐙 GitHub: [PrismaDana94](https://github.com/PrismaDana94)

---

✨ Feel free to fork & explore this repo!  
Jika ada pertanyaan atau saran, silakan open issue di repo ini.

