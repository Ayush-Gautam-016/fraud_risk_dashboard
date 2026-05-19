# 📊 Financial Risk Analytics Dashboard

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://YOUR-APP-NAME.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![Plotly](https://img.shields.io/badge/Charts-Plotly-3D4DB7?style=for-the-badge&logo=plotly)](https://plotly.com)
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458?style=for-the-badge&logo=pandas)](https://pandas.pydata.org)

> An interactive financial risk analytics dashboard that simulates real-time fraud detection, customer risk segmentation, and transaction monitoring — built with Streamlit and Plotly.

---

## 🚀 Live Demo

👉 **[Try it here → (https://ayush-gautam-fraud-risk-dashboard.streamlit.app/)**

No login needed. Filters update all charts and metrics instantly.

---

## 📸 Dashboard Features

| Feature | Description |
|--------|-------------|
| 🔴 KPI Cards | Total transactions, fraud cases, avg fraud probability, high-risk count |
| 📉 Fraud Distribution | Histogram of fraud probability scores with 0.55 threshold line |
| 📦 Risk Segmentation | Box plot of risk scores across Low / Medium / High risk tiers |
| 🔵 Scatter Analysis | Transaction amount vs fraud probability, coloured by segment |
| 🏪 Category Breakdown | Horizontal bar chart — fraud cases by merchant category |
| 🧾 Transaction Table | Full sortable, scrollable table of all filtered records |
| ⬇️ CSV Export | Download the filtered dataset as a report with one click |
| 🧠 Risk Insights | Auto-generated plain-English summary — top fraud category, avg amounts, threshold hits |

---

## 🎛️ Sidebar Filters (all live, no page reload)

- **Customer Risk Tier** — filter by Low / Medium / High Risk
- **Fraud Probability Range** — slider from 0.00 to 1.00
- **Max Transaction Amount** — up to £5,000
- **Merchant Category** — Retail, Online, Travel, Dining, ATM, Utilities

---

## 🧠 How Risk is Calculated

```
Transaction Amount  ──┐
                       ├──► Risk Score (0–100)
Fraud Probability   ──┘     = 60% fraud signal + 40% amount signal

Risk Score 0–30   →  Low Risk
Risk Score 30–65  →  Medium Risk
Risk Score 65–100 →  High Risk

Fraud Flag → triggered when Fraud Probability > 0.55
```

The dataset is **synthetic but realistic** — 500 transactions generated with `numpy` using real-world distributions (exponential spend patterns, beta-distributed fraud probabilities). `np.random.seed(42)` ensures the same data every run, making it fully reproducible for demos.

---

## 📁 Project Structure

```
fraud_risk_dashboard/
├── app.py              # Full Streamlit dashboard (single file)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Ayush-Gautam-016/fraud_risk_dashboard.git
cd fraud_risk_dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Streamlit** | Web app framework — UI, sidebar, layout |
| **Plotly Express** | Interactive charts (histogram, box, scatter, bar) |
| **Pandas** | Data manipulation and filtering |
| **NumPy** | Synthetic data generation with realistic distributions |

---

## 📊 Key Metrics Explained

**Fraud Probability** — a score from 0.0 to 1.0 representing how likely a transaction is fraudulent. Generated using a Beta distribution: most transactions cluster near 0 (safe), ~10% cluster near 1 (suspicious).

**Risk Score** — a composite business score from 0 to 100 combining fraud probability (60% weight) and normalised transaction amount (40% weight). Used to segment customers into tiers.

**Fraud Flag** — binary label applied when Fraud Probability > 0.55. This threshold can be adjusted in a real deployment based on precision/recall trade-offs.

---

## 👤 Author

**Ayush Gautam**
- GitHub: [@Ayush-Gautam-016](https://github.com/Ayush-Gautam-016)
- LinkedIn: [linkedin.com/in/ayush-gautam-89098623b](https://linkedin.com/in/ayush-gautam-89098623b)

---

## 📄 License

This project is for academic and portfolio purposes.
