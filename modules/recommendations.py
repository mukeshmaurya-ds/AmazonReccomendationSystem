import os
import pandas as pd


# --------------------------------------------------
# 1. LOAD CLEANED DATASETS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORT_DIR, exist_ok=True)


def load_data():

    products = pd.read_csv(
        os.path.join(DATA_DIR, "cleaned_products.csv")
    )

    customers = pd.read_csv(
        os.path.join(DATA_DIR, "cleaned_customers.csv")
    )

    interactions = pd.read_csv(
        os.path.join(DATA_DIR, "cleaned_interactions.csv")
    )

    return products, customers, interactions


# --------------------------------------------------
# 2. POPULAR PRODUCT RECOMMENDATIONS
# --------------------------------------------------

def get_popular_products(products, interactions, top_n=10):

    purchase_data = interactions[
        interactions["event_type"] == "purchase"
    ]

    popular = (
        purchase_data.groupby("product_id")
        .size()
        .reset_index(name="purchase_count")
    )

    popular = popular.merge(
        products,
        on="product_id",
        how="left"
    )

    popular = popular.sort_values(
        by="purchase_count",
        ascending=False
    )

    return popular[
        [
            "product_id",
            "product_name",
            "category",
            "price",
            "purchase_count"
        ]
    ].head(top_n)


# --------------------------------------------------
# 3. CUSTOMER PREFERENCE ANALYSIS
# --------------------------------------------------

def get_customer_preferences(customer_id, interactions):

    customer_data = interactions[
        interactions["customer_id"] == customer_id
    ].copy()

    if customer_data.empty:
        return pd.DataFrame(
            columns=["category", "preference_score"]
        )

    # Assign statistical weights to customer activities.
    event_weights = {
        "view": 1,
        "cart": 3,
        "purchase": 5
    }

    customer_data["activity_weight"] = (
        customer_data["event_type"]
        .map(event_weights)
        .fillna(0)
    )

    preferences = (
        customer_data.groupby("product_category")[
            "activity_weight"
        ]
        .sum()
        .reset_index()
    )

    preferences.columns = [
        "category",
        "preference_score"
    ]

    preferences = preferences.sort_values(
        by="preference_score",
        ascending=False
    )

    return preferences


# --------------------------------------------------
# 4. PERSONALIZED PRODUCT RECOMMENDATIONS
# --------------------------------------------------

def get_customer_recommendations(
    customer_id,
    products,
    interactions,
    top_n=10
):

    customer_data = interactions[
        interactions["customer_id"] == customer_id
    ].copy()

    if customer_data.empty:
        return pd.DataFrame()

    # Calculate customer category preferences.
    preferences = get_customer_preferences(
        customer_id,
        interactions
    )

    if preferences.empty:
        return pd.DataFrame()

    # Calculate customer activity weights.
    event_weights = {
        "view": 1,
        "cart": 3,
        "purchase": 5
    }

    customer_data["activity_weight"] = (
        customer_data["event_type"]
        .map(event_weights)
        .fillna(0)
    )

    # Calculate customer interest in individual products.
    product_interest = (
        customer_data.groupby("product_id")[
            "activity_weight"
        ]
        .sum()
        .reset_index()
    )

    # Products already purchased by this customer.
    purchased_products = set(
        customer_data.loc[
            customer_data["event_type"] == "purchase",
            "product_id"
        ]
    )

    # Calculate product popularity.
    popularity = (
        interactions[
            interactions["event_type"] == "purchase"
        ]
        .groupby("product_id")
        .size()
        .reset_index(name="purchase_count")
    )

    # Merge product information.
    recommendations = products.merge(
        popularity,
        on="product_id",
        how="left"
    )

    recommendations["purchase_count"] = (
        recommendations["purchase_count"].fillna(0)
    )

    # Merge customer product interest.
    recommendations = recommendations.merge(
        product_interest,
        on="product_id",
        how="left"
    )

    recommendations["activity_weight"] = (
        recommendations["activity_weight"].fillna(0)
    )

    # Merge category preferences.
    recommendations = recommendations.merge(
        preferences,
        left_on="category",
        right_on="category",
        how="left"
    )

    recommendations["preference_score"] = (
        recommendations["preference_score"].fillna(0)
    )

    # Statistical recommendation score.
    recommendations["recommendation_score"] = (
        recommendations["preference_score"] * 0.5
        + recommendations["purchase_count"] * 0.3
        + recommendations["activity_weight"] * 0.2
    )

    # Exclude products already purchased.
    recommendations = recommendations[
        ~recommendations["product_id"].isin(
            purchased_products
        )
    ]

    # Sort recommendations by score.
    recommendations = recommendations.sort_values(
        by="recommendation_score",
        ascending=False
    )

    return recommendations[
        [
            "product_id",
            "product_name",
            "category",
            "price",
            "purchase_count",
            "recommendation_score"
        ]
    ].head(top_n)


# --------------------------------------------------
# 5. FREQUENTLY BOUGHT TOGETHER ANALYSIS
# --------------------------------------------------

def get_frequently_bought_together(
    product_id,
    products,
    interactions,
    top_n=5
):

    purchases = interactions[
        interactions["event_type"] == "purchase"
    ].copy()

    # Keep only orders containing the selected product.
    selected_orders = purchases.loc[
        purchases["product_id"] == product_id,
        "order_id"
    ].dropna().unique()

    if len(selected_orders) == 0:
        return pd.DataFrame(
            columns=[
                "product_id",
                "product_name",
                "category",
                "together_count"
            ]
        )

    # Find all products in those orders.
    related_products = purchases[
        purchases["order_id"].isin(selected_orders)
    ]

    # Count other products appearing in the same orders.
    related_products = related_products[
        related_products["product_id"] != product_id
    ]

    results = (
        related_products.groupby("product_id")
        .size()
        .reset_index(name="together_count")
    )

    results = results.merge(
        products,
        on="product_id",
        how="left"
    )

    results = results.sort_values(
        by="together_count",
        ascending=False
    )

    return results[
        [
            "product_id",
            "product_name",
            "category",
            "together_count"
        ]
    ].head(top_n)


# --------------------------------------------------
# 6. SAVE POPULAR PRODUCTS REPORT
# --------------------------------------------------

def save_popular_products(products, interactions):

    popular = get_popular_products(
        products,
        interactions,
        top_n=10
    )

    output_path = os.path.join(
        REPORT_DIR,
        "popular_products.csv"
    )

    popular.to_csv(output_path, index=False)

    print("\nPopular products report saved:")
    print(output_path)


# --------------------------------------------------
# 7. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("\n===================================")
    print(" RETAILSENSE RECOMMENDATION SYSTEM ")
    print("===================================")

    products, customers, interactions = load_data()

    print("\nDataset loaded successfully!")

    print("\nTotal Products:", len(products))
    print("Total Customers:", len(customers))
    print("Total Interactions:", len(interactions))

    # Popular products.
    print("\n1. TOP POPULAR PRODUCTS")

    popular = get_popular_products(
        products,
        interactions
    )

    print(popular.to_string(index=False))

    # Customer recommendations.
    print("\n2. CUSTOMER PRODUCT RECOMMENDATIONS")

    if not customers.empty:

        selected_customer = customers.iloc[0]["customer_id"]

        print("\nCustomer ID:", selected_customer)

        preferences = get_customer_preferences(
            selected_customer,
            interactions
        )

        print("\nCustomer Category Preferences:")
        print(preferences.to_string(index=False))

        recommendations = get_customer_recommendations(
            selected_customer,
            products,
            interactions
        )

        print("\nRecommended Products:")
        print(recommendations.to_string(index=False))

    # Frequently bought together.
    print("\n3. FREQUENTLY BOUGHT TOGETHER")

    if not products.empty:

        selected_product = products.iloc[0]["product_id"]

        print("\nSelected Product ID:", selected_product)

        together = get_frequently_bought_together(
            selected_product,
            products,
            interactions
        )

        if together.empty:
            print("No related purchase products found.")
        else:
            print(together.to_string(index=False))

    # Save report.
    save_popular_products(
        products,
        interactions
    )

    print("\n===================================")
    print(" RECOMMENDATION ANALYSIS COMPLETED ")
    print("===================================")


if __name__ == "__main__":
    main()