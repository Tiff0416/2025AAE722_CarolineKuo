# Bank Churn Prediction & Customer Segmentation

End-to-end Machine Learning + SQL Pipeline + SHAP Explainability + Tableau Dashboard

This project analyzes credit card customer churn using the BankChurners dataset.
It covers SQL-based data cleaning, exploratory analysis, K-Means segmentation, churn modeling (Logistic Regression, Random Forest, XGBoost), and customer-level explainability via SHAP.
The goal is to produce actionable business recommendations supported by data.

---

# Project Structure
```
BankChurners-Project/
│
├── churn_dashboard/
│ └── app.py # Streamlit ROI simulator
│
├── data/
│ ├── raw/ # Original dataset (not uploaded to GitHub)
│ │ └── BankChurners.csv
│ │
│ └── processed/ # Cleaned + analysis-generated artifacts
│ ├── cleaned_data.csv # Cleaned dataset from SQL/Python pipeline
│ ├── bank_churners_clean.csv # Alternative cleaned version (SQL export)
│ ├── cluster_summary.csv # Summary stats for each KMeans cluster
│ ├── cluster_table.csv # Full cluster assignment table
│ ├── shap_summary.csv # SHAP feature importance summary
│ ├── roi_simulation_data.csv # ROI scenario curve used in dashboard
│ ├── X_ready.csv # Model-ready feature matrix
│ └── y_ready.csv # Model-ready target vector
│
├── notebooks/
│ ├── 01_EDA.ipynb
│ ├── 02_feature_engineering.ipynb
│ ├── 03_modeling.ipynb
│ └── 04_segmentation.ipynb
│
├── sql/
│ ├── 01_import_data.sql
│ └── etl_bankchurners.sql
│
├── src/
│ └── functions.py # Helper functions for modeling & plotting
│
├── report/
│ ├── figures/
│ └── presentation/
│
├── FinalProject_Presentation.pdf
└── README.md
```
---

# Data Cleaning (SQL Server)
1. A full ETL pipeline was built in SQL Server:

2. Loaded raw CSV into staging (stg.BankChurners_raw)

3. Standardized missing values (“Unknown” → NULL)

4. Removed duplicate CLIENTNUM

5. Cleaned categorical fields (education, marital status, income category)

6. Loaded cleaned data into ODS (ods.BankChurners)

7. Exported final processed dataset (bank_churners_clean.csv)

---
# Exploratory Data Analysis (EDA)

## Key findings:

- Churn customers show much lower transaction counts and transaction amounts

- High churn is associated with higher inactivity months

- Low credit-limit customers generally have higher churn probability

- Frequent contact with customer service is correlated with churn

## Dataset features: 46

---
# Churn Prediction Modeling

## Three models were trained:

- Logistic Regression

- Random Forest

- XGBoost (with imbalance handling: scale_pos_weight=5.22)
## Handling Class Imbalance

The dataset is highly imbalanced (Negative: 6799 vs Positive: 1302).
For XGBoost, the weighting parameter:
```
scale_pos_weight = 5.22
```
was applied to give more importance to positive class (churners) while training.

## Model Performance (Test Set)
| Model               | Accuracy  | Recall    | F1        | ROC-AUC   |
| ------------------- | --------- | --------- | --------- | --------- |
| Logistic Regression | 0.869     | 0.846     | 0.674     | 0.938     |
| Random Forest       | 0.956     | 0.825     | 0.858     | 0.987     |
| **XGBoost**         | **0.968** | **0.926** | **0.903** | **0.993** |

## Best Model: XGBoost
XGBoost significantly outperforms other methods in recall (identifying churners), F1, and ROC-AUC.

---
# SHAP Explainability (XGBoost)
SHAP was used to interpret individual predictions.

## Top Features Influencing Churn
1. Total_Trans_Ct

2. Total_Trans_Amt

3. Total_Revolving_Bal

4. Total_Ct_Chng_Q4_Q1

5. Total_Relationship_Count

6. Months_Inactive_12_mon

7. Contacts_Count_12_mon

## Patterns observed:

- Low transaction counts strongly drive churn probability upward

- Inactive months and high contact frequency indicate customers at risk

- Higher revolving balance and higher engagement reduce churn likelihood
---
# Customer Segmentation (K-Means, k=4)

## egmentation variables included:
Total_Trans_Ct, Total_Trans_Amt, Total_Revolving_Bal,
Credit_Limit, Total_Ct_Chng_Q4_Q1,
Months_Inactive_12_mon, Contacts_Count_12_mon,
Total_Relationship_Count

## Cluster Summary
| Cluster | n_customers                       | churn_rate | Key Traits                                                     |
| ------- | --------------------------------- | ---------- | -------------------------------------------------------------- |
| **0**   | 1234                              | 0.160      | Medium activity, mid credit limit, moderate churn              |
| **1**   | 4577                              | 0.041      | High activity, lowest churn group, stable loyal customers      |
| **2**   | 3174                              | 0.376      | **Highest churn-ratio**, low transaction activity, low balance |
| **3**   | 1142                              | 0.040      | Very high spenders, high credit limit, extremely low churn     |

## Intervention Target Selection

Although Cluster 2 has the highest churn rate, the extremely low activity level and high predicted cost to reactivate these customers make them a poor economic target.

Thus, this project focuses on Cluster 0 as the primary intervention group, because:

Medium churn rate (16%)

Reasonable spending / engagement baseline
---
# Business Recommendations

This project combines segmentation and SHAP insights to design targeted churn–reduction actions.
Although Cluster 2 shows the highest churn rate (37.6%), their extremely low activity and high predicted reactivation cost make them an inefficient intervention target.

## Selected Intervention Segment: Cluster 0
Cluster 0 shows:

- Moderate churn rate (16%)

- Meaningful transaction activity (avg. 58 transactions)

- Reasonable credit limits

- High potential for profitable behavior change

Intervening in Cluster 0 yields **higher ROI per dollar spent** compared to high-risk, low-engagement segments.

## Recommendation — $10 Cashback Activity Booster 
Since the dashboard assumes a $10 campaign cost, the main intervention must revolve around a $10 incentive.

### Program Design:
- Provide a one-time $10 cashback for customers who:
    make at least 3 additional transactions in 30 days, or maintain weekly activity for 4 consecutive weeks

### Why it’s aligned:
- Cost = $10 → exactly the Campaign Cost in the dashboard

- Targets Total_Trans_Ct (the top SHAP driver)

- Actionable for Cluster 0 (medium activity → high responsiveness)

### Expected lift:
- Based on benchmarking & your scenario curve: 
    3–6% retention lift → matches the dashboard’s ROI scenario.

## This section links recommendations directly to the ROI Simulator assumptions:

- Campaign cost per customer = $10

- Value per retained customer = $300

- Target segment = Cluster 0 (n = 1,234 customers)

### Under the dashboard’s assumptions, a 4.5% retention lift results in:

- 56 incremental retained customers

- Net profit = $4,319

- ROI = 0.35x

The recommendations below are designed specifically to achieve a retention lift within the 3–6% range, consistent with the ROI curve displayed on the dashboard.

---
# ROI Interpretation (Directly Reflecting the Dashboard)

## Live Demo (Streamlit App)

Interactive ROI Simulator & Churn Dashboard:

https://bank-churn-end-to-end-ml-5pqiiyrcbuujqg97zl6wi7.streamlit.app/

Streamlit simulator shows:

With Retention Lift = 4.5%,
→ Incremental customers = 56
→ Net Profit = $4,319
→ ROI = 0.35x

This matches the expected lift generated from the recommended $10 interventions, and visually aligns with your ROI vs. Retention Lift Curve plot.

---
# How to Run the Project
Clone the repository

- Execute SQL scripts in sql/ to generate processed dataset

- Run notebooks in order:

    1. 01_data_exploration.ipynb

    2. 02_feature_engineering.ipynb

    3. 03_modeling.ipynb

    4. 04_segamentation.ipynb

(Optional) Open dashboard for ROI visualization.

---
# Tech Stack
- Python (Pandas, NumPy, Scikit-Learn, XGBoost, SHAP)

- SQL Server (ETL, ODS Pipeline)

- Tableau

- Matplotlib / Seaborn

- Git & GitHub

---
# License
MIT License