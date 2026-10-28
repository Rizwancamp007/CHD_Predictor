# ❤️ Cardiovascular Disease Prediction using ML

> **Framingham Heart Study — 10-Year Coronary Heart Disease (CHD) Risk Prediction**

---

## 📌 Project Overview

The goal of this project is to build a predictive model that estimates the probability of a patient developing **Coronary Heart Disease (CHD)** over a 10-year period. Using the historical **Framingham Heart Study** dataset, we bridge the gap between raw medical data and actionable clinical insights.

### The Problem

In medical diagnostics, missing a sick patient (**False Negative**) is significantly more dangerous than a false alarm (**False Positive**). Standard Machine Learning models often fail here because they prioritize overall accuracy over **Recall** — i.e., catching the disease.

---

## 📊 Data & Features

The dataset contains records of over **4,000 patients** with **15 clinical attributes**.

### Feature Engineering: The "Pulse Pressure" Signal

We didn't just use raw data — we engineered a new feature called **Pulse Pressure (PP)**:

```
PP = Systolic BP – Diastolic BP
```

A high Pulse Pressure is a clinically proven indicator of **arterial stiffness**.

### Final Feature Selection

| Category | Features |
|---|---|
| Demographics | Age, Sex (Is_Male) |
| Behavioral | Cigarettes per Day |
| Vitals | Total Cholesterol, Systolic BP, Diastolic BP, BMI, Glucose |
| Engineered | Pulse Pressure |

---

## 🔍 Exploratory Data Analysis (EDA)

Before modeling, we analyzed the "shape" of the data to uncover hidden patterns.

- **Class Imbalance:** Only ~15% of patients in the dataset actually had CHD — a classic imbalanced class problem.
- **Missing Data:** Gaps in glucose and cholesterol levels were resolved using **Median Imputation**.

---

## 🤖 Modeling Strategy

We chose **Logistic Regression** because, in medicine, **interpretability** matters more than a black-box prediction.

### Handling Imbalance

We used the `class_weight = 'balanced'` parameter to force the model to pay more attention to **At-Risk** patients during training.

### Evaluation Metrics

| Metric | Description |
|---|---|
| **ROC-AUC** | Measures how well the model distinguishes Healthy vs. At-Risk patients |
| **Recall (Sensitivity)** | Percentage of actual sick patients correctly identified |

---

## 🎯 Threshold Tuning (Youden's J)

Most models use a 0.50 cutoff. We found this was too high for heart disease prediction.

We used **Youden's J Statistic** to find the mathematical "Sweet Spot" on the ROC Curve — the optimal decision threshold that maximizes both sensitivity and specificity.

---

## 🚀 Deployment — The Streamlit App

The final model was deployed as a web application, allowing a doctor or student to input vitals and receive an instant **Diagnostic Report**.

### App Features

- 🟢🟡🔴 **Dynamic Risk Tiers** — Labels patients as **Low**, **Moderate**, or **High** risk
- ⚠️ **Clinical Alerts** — Highlights dangerous vitals (e.g., BP > 140) in red
- 🧠 **Explainable AI** — Shows the Top 3 factors driving the patient's specific risk score (e.g., Age, Smoking)

---

## ✅ Conclusion

By combining medical knowledge (Pulse Pressure) with advanced ML techniques (Threshold Tuning), we created a tool that is far more useful than a standard classifier.

### Key Achievements

- Achieved a high **ROC-AUC** score
- Increased **Recall by ~20%** through threshold optimization
- Built a production-ready application for real-time risk assessment

---

## 🚀 Run it locally

```bash
pip install streamlit pandas numpy joblib scikit-learn
streamlit run app.py
```

## 📚 References

- Framingham Heart Study Research Data
- Scikit-Learn Documentation: Logistic Regression and Class Weights
- Clinical Guidelines for Hypertension and CHD Risk

---

<p align="center">Built with ☕ & 🤍 by Rizwan Khan</p>
