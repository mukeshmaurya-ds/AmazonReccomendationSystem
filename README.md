# RetailSense

## Data Science-Based Retail Product Recommendation & Customer Analytics System Using Python

### 1. Project Overview

RetailSense is a Data Science-based retail analytics application developed using Python. It analyses retail product information, customer demographics and shopping interactions to identify product trends, understand customer behaviour and generate rule-based product recommendations.

The application provides an interactive dashboard for exploring retail datasets and visualizing analytical results.

The project uses synthetic retail datasets for academic demonstration purposes.

### 2. Project Objectives

* Generate and analyse retail datasets.
* Clean and preprocess raw data.
* Perform Exploratory Data Analysis (EDA).
* Understand customer shopping behaviour.
* Identify popular products and categories.
* Generate rule-based product recommendations.
* Analyse product and category performance.
* Present analytical results through an interactive dashboard.

### 3. Technologies Used

| Technology | Purpose                        |
| ---------- | ------------------------------ |
| Python     | Programming language           |
| Pandas     | Data manipulation and analysis |
| NumPy      | Numerical operations           |
| Matplotlib | Data visualization             |
| Seaborn    | Statistical visualization      |
| Streamlit  | Interactive dashboard          |
| CSV        | Dataset storage                |

### 4. Main Features

1. Data Generation
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis
4. Product Popularity Analysis
5. Customer Preference Analysis
6. Rule-Based Product Recommendations
7. Frequently Bought Together Analysis
8. Customer Activity Analysis
9. Rule-Based Customer Segmentation
10. Product Performance Analysis
11. Category Performance Analysis
12. Interactive Dashboard
13. CSV Report Downloads

### 5. Project Structure

```text
RetailSense/
│
├── data/
│   ├── products.csv
│   ├── customers.csv
│   ├── interactions.csv
│   └── processed/
│
├── modules/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── recommendations.py
│   └── advanced_analytics.py
│
├── reports/
│   └── eda_charts/
│
├── requirements.txt
├── generate_data.py
├── check_data.py
├── app.py
└── README.md
```

### 6. Installation and Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### 7. Run the Project

Generate datasets:

```bash
python generate_data.py
```

Check datasets:

```bash
python check_data.py
```

Clean datasets:

```bash
python -m modules.data_cleaning
```

Run Exploratory Data Analysis:

```bash
python -m modules.eda
```

Generate product recommendations:

```bash
python -m modules.recommendations
```

Run advanced analytics:

```bash
python -m modules.advanced_analytics
```

Start the dashboard:

```bash
python -m streamlit run app.py
```

### 8. Dataset Information

The project uses three synthetic datasets:

* Products: product identifiers, names, categories, prices, stock and brands.
* Customers: customer identifiers, names, cities, ages and registration dates.
* Interactions: event identifiers, order identifiers, customer identifiers, product identifiers, event types and event dates.

### 9. Recommendation Methodology

RetailSense uses predefined statistical and business rules to generate product recommendations.

Customer preferences are calculated using weighted shopping activities:

* Product View: 1 point
* Add to Cart: 3 points
* Purchase: 5 points

Product recommendations combine category preferences, purchase popularity and customer activity scores.

This is a rule-based recommendation approach, not a Machine Learning model.

### 10. Limitations

* The project uses synthetic data rather than actual retail company data.
* Sales values are estimates based on one unit per purchase event at the listed product price.
* Actual transaction quantities, discounts and verified revenue are unavailable.
* Recommendations are generated using predefined rules.
* The application currently runs locally unless separately deployed.

### 11. Conclusion

RetailSense demonstrates how Python-based Data Science techniques can be applied to retail datasets. It provides data cleaning, exploratory analysis, customer behaviour analysis, product recommendations and interactive visualizations in a single application.

The project demonstrates the practical use of data analysis and visualization for understanding retail business activities.
