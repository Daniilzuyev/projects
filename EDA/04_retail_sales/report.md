# 🏡 Ames Housing — EDA Report

This report summarizes the results of the Exploratory Data Analysis (EDA) performed on the Ames Housing dataset.

---

## 1. Objective

Understand the main drivers of house prices in Ames, Iowa  
and prepare the dataset for machine learning models.

---

## 2. Data Cleaning Summary

### ✔ Missing values handled:
- `lot_frontage` → median  
- `garage_yr_blt` → replaced with `year_built`  
- `mas_vnr_area` → 0  
- `bsmt_qual`, `bsmt_cond`, `fireplace_qu`, `garage_qual`, `garage_cond` → `"None"`

### ✔ Cleaned fields:
- Removed duplicates  
- Normalize column names to `snake_case`  
- Converted categorical fields to `category`  

Dataset is now consistent and ML-ready.

---

## 3. Key Findings & Insights

### 🎯 3.1 Target Distribution — SalePrice

SalePrice is right-skewed.  
The majority of houses lie between **120k – 200k USD**.  
Few luxury homes reach **450k – 750k+ USD**.

The log-transformed distribution is nearly normal → good for linear models.

---

### 🏠 3.2 Living Area vs SalePrice

`GrLivArea` is strongly correlated with price.  
Larger homes → higher prices.  
Values > 4000 sq ft show variability → potential luxury segment.

---

### 🧱 3.3 Overall Quality vs Price

The strongest categorical predictor.

- Homes rated 8–10 cost **2–3×** more than houses rated 3–4.  
- Quality is a near-monotonic predictor of price.

---

### 🔥 3.4 Correlation Analysis

Top correlated features:
1. Overall Qual  
2. GrLivArea  
3. Garage Cars / Area  
4. TotalBsmtSF  
5. Year Built  
6. 1stFlrSF  

These features will form the backbone of ML models.

---

### 🗺 3.5 YearBuilt × YrSold

- Newer homes consistently sell at higher prices  
- Peak values observed for homes built after 2000  

---

## 4. ML Implications

Based on the analysis:
- A log-transformed target is recommended  
- Strong predictors suggest good performance for linear + tree models  
- Dataset is ready for feature engineering  
- No major outliers remain except luxury homes (expected)  

---

## 5. Next Steps

1. Feature engineering  
2. Build baseline models  
3. Try Ridge/Lasso + RandomForest/XGBoost  
4. Evaluate RMSE and MAE on train/validation  
5. Build final model + SHAP interpretation  

---

## 6. Conclusion

The Ames Housing dataset demonstrates clear, intuitive  
relationships between pricing and quality, size, and recency of construction.

The data is fully cleaned and ready for ML modeling.

---
