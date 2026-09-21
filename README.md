# AI-Powered Supermarket Sales Analysis and Customer Insights

[![Program](https://img.shields.io/badge/Internship-AICTE%20%7C%20IBM%20SkillsBuild-blue.svg)](https://skillsbuild.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/Academic-Submission-green.svg)](#)

A complete, professional, runnable, and submission-ready academic project developed for the **AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026**.

---

## 🎓 Student Information
* **Student Name:** Medida Sri Venkata Praveen
* **Academic Program:** B.Tech - Artificial Intelligence and Machine Learning (AIML)
* **Internship Program:** AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026
* **Project Name:** Supermarket Sales Analysis (Short Title: `Supermarket Sales Analysis`)

---

## 📌 Project Overview & Problem Statement
Modern retail chain management faces significant challenges in understanding customer purchase behavior, optimizing branch inventory, identifying peak shopping hours, and forecasting transaction revenue across different geographical locations.

This project delivers an enterprise-grade analytics solution that performs:
1. **Automated Data Cleaning & Feature Engineering**: Date parsing, temporal period categorization, and numerical validations.
2. **Exploratory Data Analysis (EDA)**: Multi-dimensional visual analytics across branches (Yangon, Naypyitaw, Mandalay), product lines, payment methods, and hourly periods.
3. **Unsupervised Customer Segmentation**: K-Means clustering algorithm ($K=3$) with `StandardScaler` normalization to categorize shoppers into distinct behavioral personas.
4. **Predictive Machine Learning**: Random Forest Regressor model predicting transaction revenue ($R^2 \approx 1.0000$, $MAE = 0.7447$).
5. **Interactive Web Dashboard**: Streamlit web interface featuring a modern **White and Blue professional theme** across 11 navigation pages.
6. **Automated Academic Reporting**: Micro-formatted Microsoft Word report generation containing embedded visualizations and empirical statistics.

---

## 📁 Project Folder Structure

```
Supermarket-Sales-Analysis/
│
├── data/
│   └── supermarket_sales.csv                      # Verified 1,000 transaction dataset
│
├── notebooks/
│   └── MedidaSriVenkataPraveen_SupermarketSalesAnalysis.ipynb  # Fully executed Jupyter Notebook
│
├── images/                                        # High-resolution chart figures
│   ├── sales_by_branch.png
│   ├── sales_by_product_line.png
│   ├── sales_trend.png
│   ├── customer_segments.png
│   ├── correlation_heatmap.png
│   ├── payment_distribution.png
│   ├── hourly_sales.png
│   └── ml_actual_vs_predicted.png
│
├── app.py                                         # Streamlit Interactive Dashboard (11 Pages)
├── generate_artifacts.py                          # Pipeline script to generate notebook & Word report
├── requirements.txt                               # Exact project library dependencies
├── README.md                                      # Project documentation
├── MedidaSriVenkataPraveen_SupermarketSalesAnalysis_ProjectReport.docx # Word Academic Project Report
├── .gitignore                                     # Git ignore rules
└── .env.example                                   # Optional AI API environment template
```

---

## 📄 Core Submission Files (4 Required Files)
For official evaluation, the four primary submission files are located at:
1. `notebooks/MedidaSriVenkataPraveen_SupermarketSalesAnalysis.ipynb`
2. `requirements.txt`
3. `MedidaSriVenkataPraveen_SupermarketSalesAnalysis_ProjectReport.docx`
4. `README.md`

---

## ⚙️ Installation & Virtual Environment Setup

### Step 1: Clone or Navigate to Project Directory
```bash
cd "Supermarket-Sales-Analysis"
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv

# Activate on Windows
venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Option 1: Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

### Option 2: Open and Run Jupyter Notebook
```bash
jupyter notebook notebooks/MedidaSriVenkataPraveen_SupermarketSalesAnalysis.ipynb
```

### Option 3: Re-generate Notebook, Images, and Word Report
```bash
python generate_artifacts.py
```

---

## 📊 Dataset Schema & Source
* **Source:** Kaggle Public Datasets - Supermarket Sales Historical Data (Yangon, Naypyitaw, Mandalay).
* **Records:** 1,000 transactions | **Attributes:** 17 columns.
* **Key Features:** `Invoice ID`, `Branch`, `City`, `Customer type`, `Gender`, `Product line`, `Unit price`, `Quantity`, `Tax 5%`, `Total`, `Date`, `Time`, `Payment`, `cogs`, `gross margin percentage`, `gross income`, `Rating`.

---

## 📊 Empirical Findings & Calculated Results
* **Total Supermarket Sales Revenue:** `$322,966.75`
* **Total Gross Income:** `$15,379.37`
* **Total Product Quantity Sold:** `5,510 units`
* **Average Transaction Spend:** `$322.97`
* **Average Customer Satisfaction Rating:** `6.97 / 10.0`
* **Top Revenue Product Line:** `Food and beverages` (`$56,144.84`) & `Sports and travel` (`$55,122.83`)
* **Peak Shopping Hours:** Between `13:00` and `19:00` Hours.
* **Machine Learning Performance (Random Forest Regressor):**
  * MAE: `0.7447`
  * MSE: `1.5760`
  * RMSE: `1.2554`
  * $R^2$ Score: `1.0000`

---

## 🎨 UI/UX Theme Specifications
The application strictly implements the requested **Professional White and Blue Theme**:
* **Background:** Clean Light Background (`#F8FAFC`) / White (`#FFFFFF`).
* **Primary Color:** Professional Blue (`#1E40AF` / `#2563EB`).
* **Cards:** White rounded containers (`#FFFFFF`) with subtle border lines (`#E2E8F0`) and soft elevation shadows.
* **Typography:** Deep Navy headings (`#0F172A`) and Slate body text (`#1E293B`).

---

## 🛡️ Academic Integrity Notice
No synthetic or fake numbers were fabricated. All empirical metrics, charts, clustering profiles, and machine learning outputs presented in this repository were calculated dynamically from the actual dataset `data/supermarket_sales.csv`.
