| Feature Group                   | Selected Variables                                                                                                              | Why Included                                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Demographics**                | Gender, Education_Level, Income_Category, Marital_Status, Customer_Age                                                          | Clear churn differences across groups                          |
| **Tenure & Family**             | Months_on_book, Tenure_bin, Dependent_count, Has_Dependents                                                                     | Longer tenure = lower churn; dependents show churn differences |
| **Behavioral (most important)** | Total_Trans_Ct, Total_Trans_Amt, Months_Inactive_12_mon, Contacts_Count_12_mon, Total_Relationship_Count, Avg_Utilization_Ratio | Large behavioral differences directly predict churn            |
| **Behavior Change (Q4/Q1)**     | Total_Amt_Chng_Q4_Q1, Total_Ct_Chng_Q4_Q1                                                                                       | Annual decline pattern = higher churn                          |
| **Financial Variables**         | Credit_Limit, Avg_Open_To_Buy, Total_Revolving_Bal                                                                              | Related to churn, skewed, requires log transform               |
| **Unknown Categories**          | Education Unknown, Income Unknown, Marital Unknown                                                                              | Unique churn patterns in these groups                          |

---

# A. Population-level EDA（Demographics & Churn）
| Variable            | EDA Insight                                                                       | Decision                 |
| ------------------- | --------------------------------------------------------------------------------- | ------------------------ |
| **Gender**          | Female churn rate higher than male                                                | Keep (one-hot)           |
| **Education Level** | Higher education (Doctorate / PG) has highest churn; Unknown also slightly higher | Keep all categories      |
| **Income Category** | U-shaped: lowest and highest income have high churn                               | Keep + consider ordering |
| **Marital Status**  | Married has lowest churn; Unknown churn slightly higher                           | Keep                     |

## Conclusion
➡ Demographic characteristics show clear churn segmentation, so all are retained.

---

# B. Tenure & Family Context

## Findings

- Months_on_book (Tenure): longer tenure → lower churn

- Dependent_count / Has_Dependents: customers with children have slightly higher churn

## Decision
✔ Keep tenure (numeric + binned version)
✔ Keep dependent variables

---

# C. High-impact Behavioral Features（Strongest Predictors）
Boxplots show clear disengagement patterns among churned customers.
| Feature                      | Insight                                                     | Decision  |
| ---------------------------- | ----------------------------------------------------------- | --------- |
| **Total_Trans_Ct**           | Attrited customers have much lower counts                   | Must keep |
| **Total_Trans_Amt**          | Attrited customers spend significantly less                 | Must keep |
| **Months_Inactive_12_mon**   | Attrited customers are more inactive                        | Keep      |
| **Contacts_Count_12_mon**    | Attrited customers contact customer service more (friction) | Keep      |
| **Total_Relationship_Count** | Attrited customers have fewer products (lower engagement)   | Keep      |
| **Avg_Utilization_Ratio**    | Attrited customers have lower utilization                   | Keep      |

➡ Behavior features are the strongest driver of churn.
---

# D. Interaction-Level Insights（Scatter Plots）

## Scatter plots revealed cluster separation：

- High transactions + high amount → stable customers

- Low transactions + low engagement → churn cluster

- Amount/Count change (Q4/Q1) also separates groups

## Decision
✔ Keep interaction variables:
Total_Amt_Chng_Q4_Q1, Total_Ct_Chng_Q4_Q1, Total_Trans_Ct, Total_Trans_Amt, Avg_Utilization_Ratio.
---
# E. Distribution Analysis & Log Transform
Skewed variables that require log1p():
| Variable             | Reason       |
| -------------------- | ------------ |
| Total_Trans_Amt      | Right skew   |
| Total_Revolving_Bal  | Heavy tail   |
| Avg_Open_To_Buy      | Extreme skew |
| Credit_Limit         | Heavy tail   |
| Total_Amt_Chng_Q4_Q1 | Slight skew  |
| Total_Ct_Chng_Q4_Q1  | Slight skew  |

---
# F. Correlation Heatmap
Most correlated with Churn:

- Total_Trans_Ct

- Total_Ct_Chng_Q4_Q1

- Total_Revolving_Bal

- Avg_Utilization_Ratio

- Total_Trans_Amt

- Total_Amt_Chng_Q4_Q1

## Decision
✔ All retained as key predictors.
---
# G. Missing Value Audit
Although no true missing values, there are many “Unknown” entries:

| Variable        | Unknown Count | Churn Rate |
| --------------- | ------------- | ---------- |
| Education Level | 1519          | 0.169      |
| Marital Status  | 749           | 0.172      |
| Income Category | 1112          | 0.168      |

## Decision
✔ Treat “Unknown” as a separate category
✔ Do not impute (contains information)

---
# Final Feature Selection Summary
| Feature Group          | Variables                                                                            | Why                          |
| ---------------------- | ------------------------------------------------------------------------------------ | ---------------------------- |
| **Demographics**       | Gender, Education, Income, Marital, Age                                              | Clear churn differences      |
| **Tenure & Family**    | Months_on_book, Tenure_bin, Dependents                                               | Stable vs unstable customers |
| **Behavioral**         | Total_Trans_Ct/Amt, Inactive Months, Relationship Count, Contacts Count, Utilization | Strongest drivers            |
| **Behavior Change**    | Amt_Chng_Q4_Q1, Ct_Chng_Q4_Q1                                                        | Declining behavior           |
| **Financial**          | Credit_Limit, Revolving_Bal, Avg_Open_To_Buy                                         | Useful after log transform   |
| **Unknown Categories** | Education/Income/Marital Unknown                                                     | Distinct segments            |
