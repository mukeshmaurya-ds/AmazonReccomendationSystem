import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================

# 1. CONFIGURATION

# ==========================================

sns.set_theme(style="whitegrid")

os.makedirs("reports/eda_charts", exist_ok=True)

# ==========================================

# 2. LOAD CLEANED DATASETS

# ==========================================

def load_data():
    products = pd.read_csv(
        "data/processed/cleaned_products.csv"
    )

    customers = pd.read_csv(
        "data/processed/cleaned_customers.csv"
    )

    interactions = pd.read_csv(
        "data/processed/cleaned_interactions.csv"
    )

    interactions["event_date"] = pd.to_datetime(
        interactions["event_date"]
    )

    return products, customers, interactions

# ==========================================

# 3. PRODUCT CATEGORY DISTRIBUTION

# ==========================================

def category_distribution(products):
    print("\n1. PRODUCT CATEGORY DISTRIBUTION")

    category_counts = products["category"].value_counts()
    print(category_counts)

    plt.figure(figsize=(9, 7))
    plt.pie(
        category_counts,
        labels=category_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Product Category Distribution")
    plt.tight_layout()
    plt.savefig("reports/eda_charts/category_distribution.png")
    plt.close()

# ==========================================

# 4. TOP 10 POPULAR PRODUCTS

# ==========================================

def popular_products(products, interactions):
    print("\n2. TOP 10 POPULAR PRODUCTS")
    purchases = interactions[interactions["event_type"] == "purchase"]
    popularity = (
        purchases.groupby("product_id")
        .size()
        .reset_index(name="purchase_count")
    )
    result = popularity.merge(products, on="product_id", how="left")
    result = result.sort_values("purchase_count", ascending=False).head(10)
    print(result[["product_name", "purchase_count"]])

    plt.figure(figsize=(12, 6))
    sns.barplot(data=result, x="purchase_count", y="product_name")
    plt.title("Top 10 Most Purchased Products")
    plt.xlabel("Number of Purchases")
    plt.ylabel("Product")
    plt.tight_layout()
    plt.savefig("reports/eda_charts/top_products.png")
    plt.close()

# ==========================================

# 5. CUSTOMER CITY DISTRIBUTION

# ==========================================

def customer_city_analysis(customers):
    print("\n3. CUSTOMER CITY DISTRIBUTION")
    city_counts = customers["city"].value_counts()
    print(city_counts)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=city_counts.index, y=city_counts.values)
    plt.title("Customer Distribution by City")
    plt.xlabel("City")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/eda_charts/customer_cities.png")
    plt.close()

# ==========================================

# 6. CUSTOMER AGE DISTRIBUTION

# ==========================================

def customer_age_analysis(customers):
    print("\n4. CUSTOMER AGE DISTRIBUTION")
    print(customers["age"].describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(data=customers, x="age", bins=10, kde=True)
    plt.title("Customer Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig("reports/eda_charts/customer_age.png")
    plt.close()

# ==========================================

# 7. PRODUCT PRICE DISTRIBUTION

# ==========================================

def price_analysis(products):
    print("\n5. PRODUCT PRICE DISTRIBUTION")
    print(products["price"].describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(data=products, x="price", bins=20, kde=True)
    plt.title("Product Price Distribution")
    plt.xlabel("Price (INR)")
    plt.ylabel("Number of Products")
    plt.tight_layout()
    plt.savefig("reports/eda_charts/product_prices.png")
    plt.close()

# ==========================================

# 8. CUSTOMER INTERACTION DISTRIBUTION

# ==========================================

def interaction_analysis(interactions):
    print("\n6. CUSTOMER INTERACTION DISTRIBUTION")
    event_counts = interactions["event_type"].value_counts()
    print(event_counts)

    plt.figure(figsize=(9, 6))
    sns.barplot(x=event_counts.index, y=event_counts.values)
    plt.title("Customer Interaction Distribution")
    plt.xlabel("Interaction Type")
    plt.ylabel("Number of Events")
    plt.tight_layout()
    plt.savefig("reports/eda_charts/interactions.png")
    plt.close()

# ==========================================

# 9. MONTHLY CUSTOMER ACTIVITY

# ==========================================

def monthly_activity(interactions):
    print("\n7. MONTHLY CUSTOMER ACTIVITY")
    monthly = interactions.set_index("event_date").resample("ME").size()
    print(monthly)

    plt.figure(figsize=(12, 6))
    monthly.plot(marker="o")
    plt.title("Monthly Customer Activity")
    plt.xlabel("Month")
    plt.ylabel("Number of Interactions")
    plt.tight_layout()
    plt.savefig("reports/eda_charts/monthly_activity.png")
    plt.close()

# ==========================================

# 10. CATEGORY-WISE REVENUE

# ==========================================

def category_revenue(products, interactions):
    print("\n8. CATEGORY-WISE REVENUE")
    purchases = interactions[interactions["event_type"] == "purchase"]
    revenue_data = purchases.merge(
        products[["product_id", "category", "price"]],
        on="product_id",
        how="left"
    )
    revenue = (
        revenue_data.groupby("category")["price"]
        .sum()
        .sort_values(ascending=False)
    )
    print(revenue)

    plt.figure(figsize=(11, 6))
    sns.barplot(x=revenue.index, y=revenue.values)
    plt.title("Revenue by Product Category")
    plt.xlabel("Category")
    plt.ylabel("Revenue (INR)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("reports/eda_charts/category_revenue.png")
    plt.close()

# ==========================================

# 11. TOP REVENUE-GENERATING PRODUCTS

# ==========================================

def top_revenue_products(products, interactions):
    print("\n9. TOP 10 REVENUE-GENERATING PRODUCTS")
    purchases = interactions[interactions["event_type"] == "purchase"]
    revenue_data = purchases.merge(
        products[["product_id", "product_name", "price"]],
        on="product_id",
        how="left"
    )
    revenue = (
        revenue_data.groupby("product_name")["price"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    print(revenue)

    plt.figure(figsize=(12, 7))
    sns.barplot(x=revenue.values, y=revenue.index)
    plt.title("Top 10 Revenue-Generating Products")
    plt.xlabel("Revenue (INR)")
    plt.ylabel("Product")
    plt.tight_layout()
    plt.savefig("reports/eda_charts/top_revenue_products.png")
    plt.close()

# ==========================================

# 12. FREQUENTLY PURCHASED PRODUCTS

# ==========================================

def frequently_purchased(products, interactions):
    print("\n10. FREQUENTLY PURCHASED PRODUCTS")
    purchases = interactions[interactions["event_type"] == "purchase"]
    result = (
        purchases.groupby("product_id")
        .size()
        .reset_index(name="purchase_count")
        .merge(products, on="product_id", how="left")
        .sort_values("purchase_count", ascending=False)
        .head(10)
    )
    print(result[["product_name", "category", "purchase_count"]])

# ==========================================

# 13. MAIN EXECUTION

# ==========================================

def main():
    print("=" * 55)
    print(" RETAILSENSE - EXPLORATORY DATA ANALYSIS")
    print("=" * 55)

    products, customers, interactions = load_data()
    category_distribution(products)
    popular_products(products, interactions)
    customer_city_analysis(customers)
    customer_age_analysis(customers)
    price_analysis(products)
    interaction_analysis(interactions)
    monthly_activity(interactions)
    category_revenue(products, interactions)
    top_revenue_products(products, interactions)
    frequently_purchased(products, interactions)

    print("\n" + "=" * 55)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("Charts saved in reports/eda_charts/")
    print("=" * 55)

if __name__ == "__main__":
    main()
