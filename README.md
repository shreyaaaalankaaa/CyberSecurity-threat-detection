# 🛡️ ThreatSentinel — AI-Powered Cybersecurity Threat Detection

> An end-to-end ML system for detecting cybersecurity threats from network traffic data, featuring a real-time Streamlit dashboard with model evaluation and explainable insights.

---

## 🚀 Live Demo
[ThreatSentinel on Replit](https://replit.com/@shreyalankaaa/ThreatSentinel)

---

## 📌 What It Does

ThreatSentinel ingests network traffic data, engineers security-relevant features, trains multiple ML classifiers, and surfaces threat predictions through an interactive dashboard — giving security analysts real-time visibility into potential intrusions.

---

## 🧠 How It Works

```
Raw Network Data → Dataset Loader → Feature Engineering → Model Training → Evaluation → Streamlit Dashboard
```

1. **Dataset Loader** (`dataset_loader.py`) — ingests and parses raw network traffic datasets
2. **Data Processor** (`data_processor.py`) — cleans, normalises, and prepares data for ML
3. **Feature Engineer** (`feature_engineer.py`) — extracts security-relevant signals from traffic patterns
4. **Model Trainer** (`model_trainer.py`) — trains classifiers (Random Forest, XGBoost, etc.)
5. **Model Evaluator** (`model_evaluator.py`) — generates accuracy, precision, recall, F1, confusion matrix
6. **Predictor** (`predictor.py`) — serves real-time threat classifications
7. **Dashboard** (`app.py`) — Streamlit UI with live charts, metrics, and predictions

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | XX% |
| Precision | XX% |
| Recall | XX% |
| F1 Score | XX% |
| Dataset | Network intrusion traffic data |

> Replace XX with your actual scores from model_evaluator.py output.

---

## ⚙️ Tech Stack

- **ML:** Python, Scikit-learn, Pandas, NumPy
- **Dashboard:** Streamlit, Plotly
- **Feature Engineering:** Custom network traffic parsers
- **Deployment:** Replit

---

## 🗂️ Project Structure

```
CyberSecurity-threat-detection/
├── app.py                 # Streamlit dashboard entry point
├── dataset_loader.py      # Data ingestion and parsing
├── data_processor.py      # Cleaning and normalisation
├── feature_engineer.py    # Security feature extraction
├── model_trainer.py       # ML model training
├── model_evaluator.py     # Performance evaluation
├── predictor.py           # Real-time prediction engine
└── utils.py               # Shared helper functions
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/shreyaaaalankaaa/CyberSecurity-threat-detection
cd CyberSecurity-threat-detection
pip install -r requirements.txt
streamlit run app.py
```

---

## 👩‍💻 Author

**Shreya Lanka** — B.Tech Cybersecurity, Sri Indu College of Engineering & Technology  
[LinkedIn](https://www.linkedin.com/in/shreya-lanka-057a0b28a/) • [GitHub](https://github.com/shreyaaaalankaaa)
