# 📁 UCI Credit Card Default — EDA Summary
## 🎯 Project Goal

Analyze financial and behavioral patterns that influence credit card default risk using the UCI Credit Card dataset.
This EDA prepares the foundation for a future classification model predicting default next month.

## 📊 Key Findings

Defaults are imbalanced:
- Only 22% of customers default — the dataset requires class-balancing in ML.
- Payment history is the strongest predictor:
- Recent delays (PAY_0, PAY_2…) show a sharp, consistent increase in default risk.

Credit limit aligns with risk:
- Customers with higher LIMIT_BAL rarely default, while low-limit customers show high delinquency.
Age has weak predictive power:
- No strong pattern between age distribution and repayment behavior.
- Demographic variables (SEX, EDUCATION, MARRIAGE) provide minimal new signal and should be treated as secondary features.

## 🛠️ Methods Used

- Missing value cleanup & feature consolidation
- Distribution analysis (histograms, KDE)
- Target distribution check
- Categorical vs target comparison
- Correlation heatmaps for PAY_X variables
- Risk segmentation by credit limit & age

## 📌 Business Insights

- Default risk depends primarily on behavior, not demographics.
- Recent late payments are the earliest and strongest warning signals.
- Credit scoring systems should weigh PAY_X history + credit limit usage significantly higher than age or gender.