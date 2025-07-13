# 💳 Loan Fraud Detection System

An AI-powered web application that predicts whether a loan application is **fraudulent or genuine**, based on financial and personal data. Built with **Python**, **Streamlit**, and **Scikit-learn**, it uses machine learning to help financial institutions assess risk in real time.

---

## 📌 Features

- ✅ Real-time fraud prediction using a trained **Random Forest Classifier**
- 📊 **Interactive dashboard** with fraud probability **gauge chart**
- 🧠 Displays **reasons behind each prediction** (Explainable AI)
- 💡 Provides **suggestions** to reduce fraud risk
- 💼 Clean, responsive **Streamlit UI** with user-friendly input forms
- 🔁 Accepts key features like income, loan amount, credit history, and more

---

## 🚀 Technologies Used

| Component         | Tech Used                       |
|------------------|----------------------------------|
| Machine Learning | scikit-learn (RandomForest)      |
| Web Framework    | Streamlit                        |
| Visualization    | Plotly (Gauge chart)             |
| Data Handling    | Pandas, NumPy                    |
| Model Saving     | Joblib                           |

---

## 📂 Project Structure

loan_fraud_detection/
├── model.pkl # Trained ML model
├── scaler.pkl # Trained feature scaler
├── train_model.py # Script to train and save model
├── app.py # Streamlit web app
├── loan_data.csv/ # Contains train/test CSV files
└── README.md


