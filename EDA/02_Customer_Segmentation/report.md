# 👥 Customer Segmentation Analysis

**Goal:** Identify which customer groups drive the most sales and profit.

---

## 📊 Dataset
- Source: Superstore Dataset  
- ~10k orders → aggregated to 1 row per customer  
- Features: `Customer ID`, `Segment`, `Region`, `Sales`, `Profit`, `Discount`, `Order Date`

---

## 🧮 Key Metrics
- Total orders  
- Total sales & profit  
- Average discount  
- Average check  

Filtered: `total_sales > 50`, `orders > 3`

---

## 📈 Insights
- **Corporate** → highest average profit  
- **Consumer** → largest base, lower margin  
- **High discounts (>20%)** → reduce profit  
- **West region** → most profitable  
- ~15% of clients bring ~60% of total profit (Pareto)

---

## 💼 Business Meaning
Focus marketing on **Corporate/West** group and review discounts in low-margin segments.
