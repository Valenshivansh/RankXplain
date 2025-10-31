# RankXplain: Explainable AI for NIRF Institutional Rankings  
![Python](https://img.shields.io/badge/python-3.10-blue)  
![License](https://img.shields.io/badge/license-MIT-green)  
![Streamlit](https://img.shields.io/badge/Streamlit-LIVE-brightgreen)  
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)  
![SHAP](https://img.shields.io/badge/SHAP-0.42-yellow)

> **Predict NIRF rankings. Explain *why* IISc tops the list. Simulate improvements.**  
> A production-grade **ML + XAI + Interactive Analytics** platform.

---

## Live Demo  
[https://rankxplain.streamlit.app](https://rankxplain.streamlit.app)  

[![Demo Video](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://youtu.be/YOUR_VIDEO_ID)  
*Click to watch 90-sec demo*

---

## Key Features  

| Feature | Description |
|-------|-----------|
| **Predictive Modeling** | XGBoost → Predicts Overall Score (R² = 0.94) |
| **SHAP Explainability** | Shows *why* each institute ranks high/low |
| **Interactive Dashboard** | Select institute → See top 3 rank drivers |
| **What-If Simulator** | "Increase RP by 10% → Rank jumps #3 → #2" |
| **Automated Insights** | "IISc leads due to RP (42%) and GO (28%)" |

---

## Tech Stack  

| Layer | Tools |
|------|-------|
| **ML** | `scikit-learn`, `xgboost` |
| **XAI** | `shap`, `lime` |
| **App** | `streamlit`, `plotly` |
| **DevOps** | GitHub Actions, Streamlit Cloud |

---

## Quick Start  

```bash
git clone https://github.com/Valenshivansh/RankXplain.git
cd RankXplain
pip install -r requirements.txt
streamlit run app/streamlit_app.py
