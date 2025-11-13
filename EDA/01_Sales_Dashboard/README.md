# 📊 Sales Dashboard & Insights

**Exploratory Data Analysis (EDA) Project**

---

## 🎯 Project Goal

Analyze sales and profit dynamics across regions, categories, and customer segments  
to identify business patterns and opportunities for growth.

---

## 📁 Dataset

- **Name:** Superstore Dataset  
- **Source:** CSV (`data/Superstore.csv`)  
- **Records:** ~10,000 rows  
- **Fields:** Order Date, Ship Date, Category, Sub-Category, Sales, Profit, Region, Segment, Discount, etc.  

---

## 🧹 Data Cleaning & Preparation

Steps performed:
1. Standardized column names (lowercase, underscores)  
2. Converted `order_date` and `ship_date` to datetime format  
3. Removed duplicates and handled missing values  
4. Optimized data types (`category`, `float32`)  
5. Verified memory usage and dataset integrity  

---

## 📊 Analysis Overview

Key analytical tasks:
- Distribution of sales and profit across **regions** and **categories**  
- Correlation analysis between `sales`, `profit`, and `discount`  
- Identification of top-performing products and customer segments  
- Trend analysis of sales over time  

---

## 📈 Visualizations

- 🟦 **Bar Chart:** Average profit by region  
- 🔥 **Heatmap:** Correlation between sales, profit, and discount  
- 📈 **Line Plot:** Total sales over time  
- 📊 **Box Plot:** Profit distribution by category  

All visualizations are available in:  
`/output/graphs/`

---

## 💡 Business Insights

- **West region** generates the highest profit margin, while **South** underperforms.  
- **High discounts (>20%)** lead to significant profit reduction.  
- **Technology category** shows the highest average sales and lowest return variability.  
- Seasonal peaks observed in **November–December**, suggesting strong holiday impact.  

---

## 🧠 Tech Stack

| Tool | Purpose |
|------|----------|
| `Python 3.11` | Main programming language |
| `Pandas 2.x` | Data manipulation & cleaning |
| `Seaborn` / `Matplotlib` | Visualization |
| `NumPy` | Numerical operations |
| `Jupyter Notebook` | Interactive analysis |

---

## 📦 Folder Structure

