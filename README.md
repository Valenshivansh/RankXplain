# 🎓 NIRF Ranking Predictor v1

> **Author:** Shivansh Pathak  
> **Goal:** Predict and analyze the *Perception (PR)* scores from NIRF data (2018–2023) using machine learning.

---


## 🧠 Project Overview
This project combines multi-year **NIRF ranking data (2018–2023)** and applies **machine learning** to identify which institutional factors most influence the *Perception (PR)* metric.

### 📊 Core Objectives
- Combine raw CSV files into one processed dataset  
- Clean and preprocess the data  
- Engineer new composite features (Faculty, Research, Outreach indices)  
- Train and evaluate regression models to predict `PR`   
- Visualize correlations and feature importance  

---

## ⚙️ Tech Stack
- **Python** (Pandas, NumPy, Seaborn, Matplotlib)
- **scikit-learn** (RandomForest, StandardScaler, Metrics)
- **Joblib** (Model persistence)

---

## 🧩 Features
| Step | Description |
|------|--------------|
| 🧹 Data Cleaning | Handle missing and non-numeric data |
| 🧠 Feature Engineering | Add derived quality and research indices |
| 📈 Model Training | Train Random Forest model to predict PR |
| 🔍 Evaluation | R² and MAE metrics to measure performance |
| 📊 Visualization | Correlation heatmap & feature importance |
| 💾 Save Model | Export trained model and scaler for reuse |

---

## 🧾 Results Summary 
| Metric | Value |
|--------|--------|
| **R² Score** | ~0.36 |
| **MAE** | ~7.45 |

These results indicate that around **36%** of the variation in perception scores can be explained by the available institutional factors — a meaningful insight given NIRF’s subjectivity.

---

## 🚀 Future Work
- Try **XGBoost / CatBoost** for improved accuracy  
- Perform **hyperparameter tuning** using GridSearchCV  
- Build a small **Streamlit dashboard** for interactive exploration  
- Extend to predict *overall ranking scores* beyond PR  

---

> *"Data reveals the unseen structure behind prestige — this model helps quantify it."*




