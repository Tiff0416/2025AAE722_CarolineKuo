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

# Data Cleaning & ETL (SQL Server)

I built a small ETL pipeline in SQL Server to standardize the BankChurners dataset:

- **Created schemas and staging table**  
  - Ensured `stg` and `ods` schemas exist.  
  - Rebuilt `stg.BankChurners_raw` with all columns as `NVARCHAR` to safely ingest the raw CSV (including extra text columns `NB1`, `NB2`).

- **Loaded raw CSV into staging**  
  - Used `BULK INSERT` to load `BankChurners.csv` into `stg.BankChurners_raw` (skipping the header row).  
  - Verified load success with `SELECT TOP (10)` and row counts.

- **Built a typed ODS table**  
  - Created `ods.BankChurners` with proper data types (e.g., `INT` for ages and counts, `FLOAT` for credit limit and utilization, `NVARCHAR` for categorical fields).  
  - Dropped helper text columns (`NB1`, `NB2`) from the ODS layer.

- **Cleaned and transformed data while loading into ODS**  
  - Used `LTRIM/RTRIM` to remove leading/trailing spaces.  
  - Converted empty strings to `NULL` via `NULLIF(..., '')`.  
  - Safely cast numeric fields with `TRY_CAST` to prevent load failures while preserving as many records as possible.

- **Validation and export**  
  - Checked that row counts between `stg.BankChurners_raw` and `ods.BankChurners` match.  
  - Exported `ods.BankChurners` to `data/processed/bank_churners_clean.csv` for downstream EDA and modeling in Python.

---
# Exploratory Data Analysis (EDA)

We performed EDA on the cleaned BankChurners dataset (46 features) to understand churn patterns and guide feature engineering:

- **Target distribution & imbalance**  
  - Converted `Attrition_Flag` into a binary `Churn_Flag` (1 = Attrited Customer).  
  - Churn rate is around **16%**, indicating a **highly imbalanced** classification problem.  
  - This later informed our modeling choices (e.g., `class_weight="balanced"`, PR-AUC focus).

- **Demographics vs churn**  
  - Gender, age, marital status, and income show **only weak correlation** with churn.  
  - Higher education segments (post-graduate, doctorate) and some high/low income groups show slightly higher churn, but overall **behavioral features matter much more** than demographics.

- **Behavioral drivers of churn**  
  - Churned customers have **much lower transaction counts and amounts**, more **inactive months**, and **fewer product relationships**.  
  - They contact customer service more frequently and have **lower utilization** of their credit limit, signaling declining engagement and growing friction before attrition.  
  - Q4/Q1 change variables (`Total_Amt_Chng_Q4_Q1`, `Total_Ct_Chng_Q4_Q1`) clearly show that **declining spending and activity are strong early-warning signals** of churn.

- **Tenure & segmentation**  
  - Binned tenure (`Months_on_book`) into ranges (e.g., `<2 years`, `2–3 years`, `3–4 years`…), and created simple flags such as `Has_Dependents`.  
  - Churn is slightly higher for longer-tenure customers and those with fewer products, but these effects are modest compared to behavioral intensity.

- **Correlation structure & feature planning**  
  - Correlation analysis confirms that **transaction count, amount, inactivity, and contacts** are the strongest predictors of churn, while demographic variables have near-zero correlation.  
  - Examined skewed numeric features (e.g., `Total_Trans_Amt`, `Credit_Limit`, `Total_Revolving_Bal`) to decide where to apply log transforms or binning in the feature engineering step.  
  - Verified missingness patterns and chose to **keep `"Unknown"` categories** (e.g., in `Education_Level`, `Marital_Status`, `Income_Category`) since their churn rates are close to the overall average.

Overall, the EDA shows that this is a **behavior-driven churn problem**: declining usage, inactivity buildup, and shallow product relationships are far more predictive than demographics. These insights directly shaped the subsequent feature engineering and modeling strategy.

---
# Feature Engineering

Feature engineering was guided directly by the findings from EDA, with the goal of enhancing predictive signal, reducing skewness, and preparing all variables for modeling.

---

## 1. Categorical Variable Encoding

EDA showed that `"Unknown"` behaves like a regular category in  
`Education_Level`, `Marital_Status`, and `Income_Category` (its churn rate is close to the dataset average).  
Therefore, no imputation or missing-indicator encoding was applied.

All categorical variables were one-hot encoded:

- Gender  
- Education_Level  
- Marital_Status  
- Income_Category  
- Card_Category  
- Tenure_bin  
- Age_bin  

This preserves full category-level information while ensuring ML compatibility.

---

## 2. Binning & Indicator Features

Based on behavioral patterns observed in EDA, several segmentation-oriented features were created:

- **Tenure bins** derived from `Months_on_book`  
  (`<2 years`, `2–3`, `3–4`, `4–5`, `5+ years`)
- **Age bins** (18–29, 30–39, 40–49, 50–59, 60+)
- **Has_Dependents** indicator (`Dependent_count > 0`)

These features help capture customer lifecycle stages and household characteristics.

---

## 3. Log Transformations for Skewed Numeric Features

EDA identified several strongly right-skewed variables, particularly:

- `Total_Trans_Amt`  
- `Avg_Open_To_Buy`  
- `Credit_Limit`  
- `Total_Revolving_Bal` (paired with a 0–1 flag)

These features were log-transformed to reduce long tails and improve model stability.

Two additional trend variables:

- `Total_Amt_Chng_Q4_Q1`  
- `Total_Ct_Chng_Q4_Q1`

were also log-transformed as an optional enhancement to support model convergence, although their skewness was less pronounced.

---

## 4. Dropping Non-Predictive Columns

Identifier and redundant descriptive columns were removed:

- `CLIENTNUM` (unique ID)
- `Attrition_Flag` (raw label)
- `Churn_Label` (text label)

Only engineered numeric features were kept for modeling.

---

## 5. Final Dataset for Modeling

The final modeling dataset includes:

- **X (features):** ~80–90 engineered variables  
- **y (target):** binary churn flag  
- **No missing values**
- **All features numeric and model-ready**

Outputs were saved as:

```bash
data/processed/X_ready.csv
data/processed/y_ready.csv
```

This feature engineering pipeline translates EDA insights into structured, machine-learning-ready inputs by emphasizing behavioral signals, correcting skewed distributions, and preserving meaningful categorical information.

---
# Modeling & Evaluation

To evaluate churn prediction performance, three supervised learning models were trained on the engineered dataset:

- Logistic Regression (with StandardScaler)
- Random Forest
- XGBoost (with imbalance-aware training)

All models were trained using an 80/20 stratified split to preserve the original churn distribution.

---

## 1. Handling Class Imbalance

The dataset is imbalanced (churn ≈ 16%).  
To address this:

- Logistic Regression and Random Forest used **class_weight="balanced"**
- For XGBoost, we computed:

```bash
scale_pos_weight = negative_count / positive_count = 5.22
```

This increases the relative weight of churners and improves recall and ROC-AUC.

---
## 2. Model Performance (Test Set)

| Model               | Accuracy  | Recall    | F1        | ROC-AUC   |
| ------------------- | --------- | --------- | --------- | --------- |
| Logistic Regression | 0.869     | 0.846     | 0.674     | 0.938     |
| Random Forest       | 0.956     | 0.825     | 0.858     | 0.987     |
| **XGBoost**         | **0.968** | **0.926** | **0.903** | **0.993** |

Key observations:

- **Logistic Regression** performs reasonably well but lacks capacity to capture nonlinear patterns.
- **Random Forest** improves performance substantially with higher F1 and ROC-AUC.
- **XGBoost** achieves the best metrics overall, particularly in:
  - **Recall** (detecting churners)
  - **F1-score** (precision–recall balance)
  - **ROC-AUC** (discrimination ability)

Thus, XGBoost is selected as the final model.

---

## 3. Model Explainability via SHAP

To understand how the model arrives at predictions, SHAP values were computed for XGBoost.  
This provides global feature importance as well as individual-level explanations.

### Top Features (Mean Absolute SHAP Values)

1. **Total_Trans_Ct** – transaction frequency  
2. **Total_Trans_Amt** – spending level  
3. **Total_Revolving_Bal** – active revolving usage  
4. **Total_Ct_Chng_Q4_Q1** – activity trend (Q4→Q1 change)  
5. **Total_Relationship_Count** – product depth  
6. **Months_Inactive_12_mon** – inactivity  
7. **Contacts_Count_12_mon** – customer-service interaction

### Key behavioral patterns:

- **Low transaction activity** (count/amount) consistently increases churn probability.  
- **Inactivity buildup** and **higher contact frequency** push churn risk upward.  
- **Stronger product engagement** and **higher revolving usage** reduce churn likelihood.  
- Demographic variables have relatively minor contributions.

These insights later guide the segmentation and intervention strategy.

---
# Customer Segmentation (K-Means, k = 4)

To translate model insights into actionable retention strategies, I performed an unsupervised segmentation based on the **behavioral features** identified by SHAP as the strongest churn drivers.

## 1. Feature selection & preprocessing

Segmentation was built on the following SHAP-based behavioral variables:

- Total_Trans_Ct  
- Total_Trans_Amt  
- Total_Revolving_Bal  
- Credit_Limit  
- Total_Ct_Chng_Q4_Q1  
- Months_Inactive_12_mon  
- Contacts_Count_12_mon  
- Total_Relationship_Count  

Steps:

- Kept only these features (plus `Churn_Flag` for profiling, not clustering).
- Standardized features using `StandardScaler` before K-Means.
- Used only numeric, non-missing columns.

---

## 2. Choosing the number of clusters

We evaluated K-Means using values of k from 2 to 8.
I fitted K-Means and computed:

- **Inertia (SSE)** – for the elbow method  
- **Silhouette score** – for cluster separation quality  

Both diagnostics suggested a good trade-off around **k = 4**, balancing compactness, separation, and interpretability.  
Thus, the final model uses **K-Means with 4 clusters** and `n_init = 20` for stability.

---

## 3. Cluster profiles

After fitting the final model, cluster labels were added back to the full dataset and summarized:

| Cluster | n_customers | churn_rate | Key Traits                                                     |
| ------- | ----------- | ---------- | -------------------------------------------------------------- |
| **0**   | 1234        | 0.160      | Mid activity, mid credit limit, moderate churn                |
| **1**   | 4577        | 0.041      | High activity, lowest churn, stable loyal customers           |
| **2**   | 3174        | 0.376      | **Highest churn**, low transactions, low balances, weak usage |
| **3**   | 1142        | 0.040      | Very high spenders, high limits, extremely low churn          |

- Cluster 1 and 3 represent **loyal, high-activity** segments with very low churn.
- Cluster 2 contains **heavily disengaged** customers with the highest churn rate but very low baseline activity.
- Cluster 0 is a **mid-activity, mid-value segment** with a churn rate (~16%) clearly above the best groups but still economically meaningful to reactivate.

All cluster-level metrics were saved to:

```bash
data/processed/cluster_summary.csv
data/processed/cluster_tableau.csv   # used as the row-level source for the Streamlit app
```
---
# Intervention target selection (Business Recommendations )
Although **Cluster 2** has the highest churn rate (~38%), these customers already exhibit **very low engagement and low balances**, implying a high cost and low probability of successful reactivation.

Instead, the project focuses on **Cluster 0** as the primary intervention group because:

- They have **moderate churn (16%)**: there is room for improvement.

- They maintain **reasonable credit limits and balances**: economic value per retained customer is non-trivial.

- Their behavioral profile matches SHAP risk patterns (inactivity ↑, transactions ↓), suggesting they are **early-stage disengagers**, not fully lost.

This makes Cluster 0 a **more promising ROI target** for proactive retention campaigns.

---
# ROI simulation for a reactivation campaign (Cluster 0)

To quantify potential business impact, I simulated a simple reactivation campaign for Cluster 0:

Assumptions (example values):

- Segment size: 1234 customers

- Baseline churn rate: 16%

- Annual value per retained customer: $300

- Offer: per-customer cashback levels 0,5,10,15,20,25,30 USD

A simple response model assumes:

Every additional $5 cashback → **1 percentage point absolute churn reduction in Cluster 0**.

For each cashback level, I computed:

- Expected churn reduction (retention lift)

- Incremental customers retained

- Incremental revenue from retention

- Campaign cost (cashback × segment size)

- **Net profit and ROI**

Results were exported for the Streamlit app:
```bash
data/processed/roi_simulation.csv
```

The app reads these files to update ROI curves interactively when the user adjusts campaign assumptions.

---
# Experimental design (proposed A/B test)

To validate the strategy in practice, I designed a simple randomized controlled experiment for Cluster 0 (not implemented in production, but used as an experiment framework):

- Randomly split Cluster 0 into Treatment vs Control (e.g., 50/50).
- Treatment: receive a targeted offer, such as \$10 cashback after 3 transactions within 30 days.
- Control: no special intervention.
- Track over 60 days:
  - Churn rate
  - Transaction count
  - Revenue per user

The Streamlit app is used to **simulate** different offer levels (cashback amounts) and visualize their expected churn reduction, incremental revenue, campaign cost, and ROI, providing a decision-support tool for designing such an experiment.
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

- Matplotlib / Seaborn

- Git & GitHub

---
# License
MIT License
