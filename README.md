#  Loan Approval Predictor

A machine learning web app that predicts whether a loan application will be approved or rejected — instantly, based on applicant and financial details.
## Live Demo:https://loan-approval-predicator.streamlit.app

---

##  Overview

This project uses a trained **Support Vector Machine (SVM)** model wrapped in a **scikit-learn Pipeline** to classify loan applications. The model is served through an interactive **Streamlit** web interface where users can fill in applicant details and get a real-time prediction with confidence score.

---

##  Project Structure

```
Loan_Approval_Prediction/
│
├── main.py                      # Streamlit web app
├── classifier.pkl               # Trained SVM pipeline (model file)
├── loan_approval.csv            # Dataset used for training
├── Loan_approval_system.ipynb   # Full EDA, training, and evaluation notebook
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

##  Setup & Installation

**1. Clone the repository**
```bash
https://github.com/Jitendra-Shah1/loan-approval-predicator.git
cd loan-approval-predictor
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run main.py
```

---

##  Model Details

| Property        | Value                          |
|----------------|-------------------------------|
| Algorithm       | Support Vector Machine (SVM)  |
| Kernel          | RBF                           |
| Preprocessing   | StandardScaler + OneHotEncoder via ColumnTransformer |
| Train/Test Split| 80% / 20%                     |
| Test Accuracy   | ~95%                          |

**Features used for prediction:**

| Feature | Type | Preprocessing |
|---|---|---|
| CIBIL Credit Score | Numerical | StandardScaler |
| Annual Income | Numerical | StandardScaler |
| Loan Amount | Numerical | StandardScaler |
| Total Assets Value | Numerical (computed) | StandardScaler |
| Loan Term | Passthrough | None |
| No. of Dependents | Passthrough | None |
| Education | Categorical | OneHotEncoder |
| Self Employed | Categorical | OneHotEncoder |

---

##  App Input Fields

| Field | Description |
|---|---|
| CIBIL Credit Score | Applicant's credit score (300–900) |
| Annual Income | Yearly income in USD |
| Education | Graduate / Not Graduate |
| Self Employed | Yes / No |
| No. of Dependents | Number of financial dependents (0–5) |
| Loan Amount | Requested loan in USD |
| Loan Term | Repayment duration in years (2–14) |
| Residential Asset Value | Value of residential property |
| Commercial Asset Value | Value of commercial property |
| Luxury Asset Value | Value of luxury items |
| Bank Asset Value | Savings / deposits |

---

##  Dataset

- **File:** `loan_approval.csv`
- **Rows:** 4,269 loan applications
- **Target column:** `loan_status` → `Approved` / `Rejected`
- **Source:** https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

---

##  Dependencies

Key libraries used:

- `streamlit` — web UI
- `scikit-learn` — ML pipeline and SVM model
- `pandas` / `numpy` — data processing
- `matplotlib` / `seaborn` — EDA visualizations (notebook only)

Full list in `requirements.txt`.

---




