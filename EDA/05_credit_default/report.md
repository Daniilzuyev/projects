# 🏦 UCI Credit Card Default — EDA Report
This report summarizes the results of the Exploratory Data Analysis (EDA) performed on the UCI Credit Card Default dataset.

## 1. Objective

Understand the key drivers of credit card default risk
and prepare the dataset for machine learning modeling.

## 2. Data Cleaning Summary
✔ Missing values handled:
limit_bal → filled with mean
education → invalid values {0,4,5,6} mapped to 3
marriage → invalid value {0} mapped to 3
pay_0–pay_6 → converted to integer
Verified no missing values in bill_amtX, pay_amtX, or target column

✔ Cleaned fields:
Removed duplicates
Normalized all column names to snake_case
Converted categorical fields (sex, education, marriage) to category
Ensured payment history fields are consistent and ML-ready
Dataset is now clean, consistent, and ready for modelling.

## 3. Key EDA Findings
Default distribution
Strong class imbalance:
78% non-default
22% default
Must be handled before training ML models.
Payment delay variables (PAY_0…PAY_6)
Strong internal correlations (0.5 – 0.8) → customers’ behaviour tends to repeat each month.
Most customers pay on time (<= 0).
Default risk rises sharply once delays reach 1–2 months.
Severe delays (3+) are rare but extremely high-risk.
Credit limit (limit_bal)
Higher limits correspond to customers with on-time payments.
Lower limits correlate with more frequent delays and higher default rates.
Age
Median age remains stable across payment groups.
No strong relationship between age and default risk.
Demographics
sex, education, marriage show small, non-significant variation.
Individually they are weak predictors.

## 4. Business Insights
Payment behaviour (PAY_X) is the strongest indicator of future default.
Even one month of delay significantly increases probability of default.
Credit limit works as an indirect indicator of customer reliability.
Demographic attributes add little predictive power on their own.