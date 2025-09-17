# Smart Loan Recovery System (Machine Learning)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-lightgrey)](https://plotly.com/python/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

A compact, end‑to‑end notebook that demonstrates how to:
1) explore loan & borrower data,  
2) segment borrowers with **K‑Means**,  
3) predict **default/recovery risk** with a **Random Forest**, and  
4) **assign the optimal recovery strategy** (legal action / settlement / reminders) based on the predicted risk.

This project is ideal as a portfolio piece: it shows business framing, data analysis, unsupervised segmentation, supervised risk modeling, and an actionable decision layer.


---

## ✨ Key Features

- **Exploratory Data Analysis (EDA)** with clean, storytelling visualizations (Plotly):
  - Loan amount distribution & income relationships
  - Payment history vs. recovery outcomes
  - Missed payments vs. recovery status
- **Unsupervised Segmentation**: `StandardScaler` + `KMeans (k=4)` to create `Borrower_Segment` and human‑readable `Segment_Name`.
- **Risk Modeling**: `RandomForestClassifier(n_estimators=100, random_state=42)` to produce a **probability of high risk** (`Risk_Score`).
- **Decision Layer**: maps `Risk_Score` to a **Recovery Strategy**:
  - `> 0.75` → **Immediate legal notices & aggressive recovery**
  - `0.50–0.75` → **Settlement offers & repayment plans**
  - `< 0.50` → **Automated reminders & monitoring**

> The notebook file: **`Smart_Loan_Recovery_System_with_Machine_Learning.ipynb`**

---

## 📊 Data Schema (expected columns)

The notebook expects a CSV named **`loan-recovery.csv`** with columns like:

| Column                       | Type      | Description                                               |
|-----------------------------|-----------|-----------------------------------------------------------|
| `Borrower_ID`               | string/int| Unique borrower identifier                                |
| `Age`                       | int       | Borrower age                                              |
| `Monthly_Income`            | float     | Monthly income (USD)                                      |
| `Loan_Amount`               | float     | Loan principal (USD)                                      |
| `Loan_Tenure`               | int       | Tenure in months                                          |
| `Interest_Rate`             | float     | Interest rate (%)                                         |
| `Collateral_Value`          | float     | Collateral valuation (USD)                                |
| `Outstanding_Loan_Amount`   | float     | Current outstanding amount (USD)                          |
| `Monthly_EMI`               | float     | Monthly installment                                       |
| `Num_Missed_Payments`       | int       | Number of missed payments                                 |
| `Days_Past_Due`             | int       | Days past due                                             |
| `Payment_History`           | category  | e.g., Good / Average / Poor                               |
| `Recovery_Status`           | category  | e.g., `Recovered` / `Not Recovered`                       |
| `Collection_Method`         | category  | e.g., Call / Email / Visit                                |
| `Collection_Attempts`       | int       | Number of attempts                                        |
| `Legal_Action_Taken`        | bool/int  | Indicator if legal steps were taken                       |

> The notebook creates: `Borrower_Segment`, `Segment_Name`, `High_Risk_Flag`, `Risk_Score`, `Predicted_High_Risk`, and `Recovery_Strategy`.

---

## 🧠 Methodology

```mermaid
flowchart LR
  A[CSV: loan-recovery.csv] --> B[EDA & Cleaning]
  B --> C[Scaling: StandardScaler]
  C --> D[KMeans (k=4) → Borrower_Segment]
  D --> E[Segment_Name mapping]
  E --> F[train_test_split (75/25 default)]
  F --> G[RandomForestClassifier → Risk_Score (P(high risk))]
  G --> H{Thresholds}
  H -->|>0.75| I[Legal action]
  H -->|0.50–0.75| J[Settlement/repayment plans]
  H -->|<0.50| K[Automated reminders]
```

- **Segmentation** highlights distinct borrower profiles (e.g., “High Income, Low Default Risk” vs “High Loan, Higher Default Risk”).  
- **Random Forest** outputs `Risk_Score` = probability of being high‑risk (`Predicted_High_Risk` uses 0.50 threshold).  
- **Decision rules** translate probabilities into **concrete actions** for collections.

---

## 🚀 Quickstart

### 1) Clone & environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2) Data
Place your dataset at the project root as **`loan-recovery.csv`** with the schema above.

### 3) Run the notebook
```bash
jupyter notebook Smart_Loan_Recovery_System_with_Machine_Learning.ipynb
```
Then execute cells top‑to‑bottom. The Plotly charts are interactive in Jupyter/Colab.

> **Tip:** To publish static visuals in the README, take a few screenshots from the notebook (e.g., distribution plots, segmentation scatter) and add them to a `/assets` folder, then embed them here.

---

## 🔎 What to Look For / Extend

- **Model quality**: add evaluation (ROC‑AUC, PR‑AUC, confusion matrix, calibration).  
- **Feature importance**: permutation importance or SHAP to explain drivers of risk.  
- **Threshold tuning**: optimize cutoffs by business costs (legal fees vs. recovery rate).  
- **Strategy simulator**: estimate expected recovery and cost per strategy.  
- **Fairness checks**: ensure no protected group is disproportionately impacted.  
- **MLOps**: move from notebook to a small API (FastAPI) + CI, unit tests, and MLflow logging.

---

## 🧰 Tech Stack

- **Python** (3.10+) • **pandas** • **scikit‑learn** • **Plotly** • **Jupyter Notebook**

---

## 📁 Suggested Repo Structure

```
.
├── Smart_Loan_Recovery_System_with_Machine_Learning.ipynb
├── loan-recovery.csv                 # not included; add your data here
├── requirements.txt
├── assets/                           # screenshots/figures for README
└── README.md
```

---

## 📜 License

MIT — free to use, change, and distribute. You can replace this with your preferred license.

---

## 🙌 Author

**Nikita Marshchonok**  
GitHub: https://github.com/NikitaMarshchonok  
LinkedIn: http://www.linkedin.com/in/nikita-marshchonok  
Email: n.marshchonok@gmail.com

---
