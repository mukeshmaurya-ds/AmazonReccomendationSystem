import os
import pandas as pd


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

os.makedirs(REPORT_DIR, exist_ok=True)


# ==================================================
# 2. LOAD CLEANED DATASETS
# ==================================================

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

    interactions["event_date"] = pd.to_datetime(
        interactions["event_date"],
        errors="coerce"
    )

    return products, customers, interactions


# ==================================================
# 3. CUSTOMER ACTIVITY ANALYSIS
# ==================================================

def customer_activity_analysis(customers, interactions):

    activity = interactions.copy()

    # Count customer activities.
    activity_summary = (
        activity.pivot_table(
            index="customer_id",
            columns="event_type",
            values="event_id",
            aggfunc="count",
            fill_value=0
        )
        .reset_index()
    )

    # Ensure all event columns exist.
    for event in ["view", "cart", "purchase"]:

        if event not in activity_summary.columns:

            activity_summary[event] = 0

    # Calculate total activities.
    activity_summary["total_activities"] = (
        activity_summary["view"]
        + activity_summary["cart"]
        + activity_summary["purchase"]
    )

    # Calculate unique products explored.
    unique_products = (
        activity.groupby("customer_id")["product_id"]
        .nunique()
        .reset_index(name="unique_products")
    )

    activity_summary = activity_summary.merge(
        unique_products,
        on="customer_id",
        how="left"
    )

    # Find the latest customer activity.
    last_activity = (
        activity.groupby("customer_id")["event_date"]
        .max()
        .reset_index(name="last_activity_date")
    )

    activity_summary = activity_summary.merge(
        last_activity,
        on="customer_id",
        how="left"
    )

    # Add customer details.
    activity_summary = activity_summary.merge(
        customers,
        on="customer_id",
        how="left"
    )

    return activity_summary


# ==================================================
# 4. CUSTOMER SEGMENTATION
# ==================================================

def customer_segmentation(activity_summary):

    result = activity_summary.copy()

    # Rule-based customer segmentation.
    # These are predefined project rules,
    # not Machine Learning predictions.

    def assign_segment(row):

        purchases = row["purchase"]

        carts = row["cart"]

        views = row["view"]

        if purchases >= 5:

            return "Loyal Customer"

        elif purchases >= 2:

            return "Regular Customer"

        elif carts >= 2:

            return "Potential Customer"

        elif views >= 3:

            return "Browsing Customer"

        else:

            return "New / Low Activity Customer"

    result["customer_segment"] = result.apply(
        assign_segment,
        axis=1
    )

    return result


# ==================================================
# 5. PRODUCT PERFORMANCE ANALYSIS
# ==================================================

def product_performance_analysis(products, interactions):

    purchases = interactions[
        interactions["event_type"] == "purchase"
    ]

    views = interactions[
        interactions["event_type"] == "view"
    ]

    carts = interactions[
        interactions["event_type"] == "cart"
    ]

    # Count product views.
    view_counts = (
        views.groupby("product_id")
        .size()
        .reset_index(name="view_count")
    )

    # Count cart activities.
    cart_counts = (
        carts.groupby("product_id")
        .size()
        .reset_index(name="cart_count")
    )

    # Count purchases.
    purchase_counts = (
        purchases.groupby("product_id")
        .size()
        .reset_index(name="purchase_count")
    )

    # Combine product statistics.
    result = products.copy()

    result = result.merge(
        view_counts,
        on="product_id",
        how="left"
    )

    result = result.merge(
        cart_counts,
        on="product_id",
        how="left"
    )

    result = result.merge(
        purchase_counts,
        on="product_id",
        how="left"
    )

    for column in [
        "view_count",
        "cart_count",
        "purchase_count"
    ]:

        result[column] = result[column].fillna(0).astype(int)

    # Estimated sales value.
    result["estimated_sales_value"] = (
        result["purchase_count"] * result["price"]
    )

    return result.sort_values(
        by="purchase_count",
        ascending=False
    )


# ==================================================
# 6. CATEGORY PERFORMANCE ANALYSIS
# ==================================================

def category_performance_analysis(
    products,
    interactions
):

    performance = product_performance_analysis(
        products,
        interactions
    )

    category_summary = (
        performance.groupby("category")
        .agg(
            total_products=("product_id", "count"),
            total_views=("view_count", "sum"),
            total_carts=("cart_count", "sum"),
            total_purchases=("purchase_count", "sum"),
            estimated_sales_value=(
                "estimated_sales_value",
                "sum"
            )
        )
        .reset_index()
    )

    return category_summary.sort_values(
        by="total_purchases",
        ascending=False
    )


# ==================================================
# 7. GENERATE CUSTOMER REPORT
# ==================================================

def generate_customer_report(
    customers,
    interactions
):

    activity = customer_activity_analysis(
        customers,
        interactions
    )

    segmented = customer_segmentation(activity)

    output_path = os.path.join(
        REPORT_DIR,
        "customer_activity_report.csv"
    )

    segmented.to_csv(
        output_path,
        index=False
    )

    print("\nCustomer report generated:")
    print(output_path)

    return segmented


# ==================================================
# 8. GENERATE PRODUCT REPORT
# ==================================================

def generate_product_report(
    products,
    interactions
):

    report = product_performance_analysis(
        products,
        interactions
    )

    output_path = os.path.join(
        REPORT_DIR,
        "product_performance_report.csv"
    )

    report.to_csv(
        output_path,
        index=False
    )

    print("\nProduct report generated:")
    print(output_path)

    return report


# ==================================================
# 9. GENERATE CATEGORY REPORT
# ==================================================

def generate_category_report(
    products,
    interactions
):

    report = category_performance_analysis(
        products,
        interactions
    )

    output_path = os.path.join(
        REPORT_DIR,
        "category_performance_report.csv"
    )

    report.to_csv(
        output_path,
        index=False
    )

    print("\nCategory report generated:")
    print(output_path)

    return report


# ==================================================
# 10. MAIN FUNCTION
# ==================================================

def main():

    print("\n======================================")
    print(" RETAILSENSE ADVANCED ANALYTICS ")
    print("======================================")

    products, customers, interactions = load_data()

    print("\nDatasets loaded successfully!")

    # Customer activity.
    print("\n1. CUSTOMER ACTIVITY ANALYSIS")

    customer_report = customer_activity_analysis(
        customers,
        interactions
    )

    print(
        customer_report[
            [
                "customer_id",
                "view",
                "cart",
                "purchase",
                "total_activities"
            ]
        ].head(10).to_string(index=False)
    )

    # Customer segmentation.
    print("\n2. CUSTOMER SEGMENTATION")

    segmented = customer_segmentation(
        customer_report
    )

    print(
        segmented["customer_segment"]
        .value_counts()
        .to_string()
    )

    # Product performance.
    print("\n3. PRODUCT PERFORMANCE")

    product_report = product_performance_analysis(
        products,
        interactions
    )

    print(
        product_report[
            [
                "product_id",
                "product_name",
                "view_count",
                "cart_count",
                "purchase_count",
                "estimated_sales_value"
            ]
        ].head(10).to_string(index=False)
    )

    # Category performance.
    print("\n4. CATEGORY PERFORMANCE")

    category_report = category_performance_analysis(
        products,
        interactions
    )

    print(category_report.to_string(index=False))

    # Generate CSV reports.
    print("\n5. GENERATING REPORTS")

    generate_customer_report(
        customers,
        interactions
    )

    generate_product_report(
        products,
        interactions
    )

    generate_category_report(
        products,
        interactions
    )

    print("\n======================================")
    print(" ADVANCED ANALYTICS COMPLETED ")
    print("======================================")


if __name__ == "__main__":

    main()