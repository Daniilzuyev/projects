# 🏡 Ames Housing — Real Estate EDA

A complete exploratory data analysis of the famous Ames Housing dataset,  
designed to understand the key drivers behind house prices and prepare the data for ML modeling.

---

## 📌 Project Goals

- Analyze the main factors that influence house prices  
- Discover relationships between quality, area, year built, and price  
- Identify outliers and distribution patterns  
- Build ML-ready cleaned dataset  
- Produce visual insights used in real-world real estate analytics  

---

## 📊 Dataset

**Ames Housing Dataset (79 features)**  
Source: open dataset widely used in ML education.

The dataset contains:
- Numerical features (area, basement, garage, lot)
- Categorical features (quality ratings, materials, neighborhood)
- Time features (year built, year remodeled, year sold)

---

## 🧼 Data Cleaning

We performed:
- Missing value imputation  
- Category normalization (`None` for missing basement/garage/fireplace info)  
- Conversion to numeric and category types  
- Duplicate removal  
- Column normalization to `snake_case`  

This makes the dataset fully **ML-ready**.

---

## 📈 Key Visualizations

- Distribution of SalePrice  
- Log-transformed price distribution  
- Living area vs SalePrice  
- Overall Quality vs SalePrice  
- Correlation matrix  
- SalePrice correlations  
- YearBuilt × YrSold heatmap  
- Pairplot  

All graphics are stored in `output/graphs/`.

---

## 🔥 Business Insights

The strongest predictors of price are:

- Overall home quality  
- Living area (GrLivArea)  
- Basement and garage size  
- Year built  
- Neighborhood  

Newer and higher-quality homes sell for significantly more.

---

## 🧠 ML Readiness

This EDA prepares the dataset for:
- Linear Regression  
- Ridge / Lasso / ElasticNet  
- Tree-based models (RF, XGBoost, LightGBM)  
- Feature engineering (log transforms, interactions, OHE)  


