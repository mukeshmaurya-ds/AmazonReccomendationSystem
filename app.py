# ============================================================
# RetailSense
# Data Science-Based Retail Product Recommendation
# & Customer Analytics System
# ============================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from modules.recommendations import (
    get_popular_products,
    get_customer_preferences,
    get_customer_recommendations,
    get_frequently_bought_together
)

from modules.advanced_analytics import (
    customer_activity_analysis,
    customer_segmentation,
    product_performance_analysis,
    category_performance_analysis
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RetailSense | Retail Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. PROFESSIONAL CSS - LIGHT THEME
# ============================================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #f4f7fb;
    color: #1e293b;
}

/* Main content area */
[data-testid="stAppViewContainer"] {
    background-color: #f4f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headings */
h1 {
    color: #172554 !important;
    font-weight: 800;
}

h2, h3 {
    color: #1e3a8a !important;
    font-weight: 700;
}

/* Normal text */
p, label, .stMarkdown {
    color: #334155;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #ffffff !important;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 3px 8px rgba(0,0,0,0.06);
}

/* Metric labels */
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #475569 !important;
    font-weight: 600 !important;
}

/* Metric values */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #172554 !important;
    font-weight: 800 !important;
}

/* Metric delta */
[data-testid="stMetricDelta"] {
    color: #15803d !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #172554 !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Sidebar navigation */
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #ffffff !important;
}

/* Buttons */
.stButton button,
.stDownloadButton button {
    border-radius: 8px;
    font-weight: 600;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: #cbd5e1;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

PRODUCTS_FILE = os.path.join(
    PROCESSED_DIR, "cleaned_products.csv"
)

CUSTOMERS_FILE = os.path.join(
    PROCESSED_DIR, "cleaned_customers.csv"
)

INTERACTIONS_FILE = os.path.join(
    PROCESSED_DIR, "cleaned_interactions.csv"
)


# ============================================================
# 4. LOAD DATASETS
# ============================================================

@st.cache_data
def load_datasets():

    required_files = [
        PRODUCTS_FILE,
        CUSTOMERS_FILE,
        INTERACTIONS_FILE
    ]

    for file_path in required_files:

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Required dataset not found: {file_path}"
            )

    products = pd.read_csv(PRODUCTS_FILE)

    customers = pd.read_csv(CUSTOMERS_FILE)

    interactions = pd.read_csv(INTERACTIONS_FILE)

    # Convert date columns
    if "registration_date" in customers.columns:
        customers["registration_date"] = pd.to_datetime(
            customers["registration_date"],
            errors="coerce"
        )

    if "event_date" in interactions.columns:
        interactions["event_date"] = pd.to_datetime(
            interactions["event_date"],
            errors="coerce"
        )

    # Convert numeric columns
    if "price" in products.columns:
        products["price"] = pd.to_numeric(
            products["price"],
            errors="coerce"
        )

    if "age" in customers.columns:
        customers["age"] = pd.to_numeric(
            customers["age"],
            errors="coerce"
        )

    return products, customers, interactions


# ============================================================
# 5. LOAD DATA SAFELY
# ============================================================

try:

    products, customers, interactions = load_datasets()

except Exception as error:

    st.error("Unable to load project datasets.")

    st.exception(error)

    st.info(
        "Run the data cleaning module first: "
        "python -m modules.data_cleaning"
    )

    st.stop()


# ============================================================
# 6. COMMON HELPER FUNCTIONS
# ============================================================

def show_empty_message(message="No data available for this analysis."):

    st.info(message)


def show_dataframe(dataframe, height=350):

    if dataframe is None or dataframe.empty:

        show_empty_message()

    else:

        st.dataframe(
            dataframe,
            use_container_width=True,
            height=height
        )


def create_bar_chart(
    dataframe,
    x_column,
    y_column,
    title,
    xlabel=None,
    ylabel=None
):

    if dataframe.empty:
        show_empty_message()
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=dataframe,
        x=x_column,
        y=y_column,
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.set_xlabel(xlabel or x_column)

    ax.set_ylabel(ylabel or y_column)

    plt.xticks(rotation=35, ha="right")

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


def download_csv(dataframe, filename, label):

    if dataframe is None or dataframe.empty:

        st.info("No report data available to download.")

        return

    csv_data = dataframe.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=label,
        data=csv_data,
        file_name=filename,
        mime="text/csv"
    )


def get_estimated_sales(interactions, products):

    purchases = interactions[
        interactions["event_type"] == "purchase"
    ].copy()

    purchases = purchases.merge(
        products[["product_id", "price"]],
        on="product_id",
        how="left"
    )

    purchases["estimated_sales"] = (
        purchases["price"].fillna(0)
    )

    return purchases


# ============================================================
# 7. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🛒 RetailSense")

st.sidebar.caption(
    "Retail Product Recommendation & Customer Analytics"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Product Analysis",
        "Customer Analytics",
        "Product Recommendations",
        "Sales Analytics",
        "Advanced Analytics & Reports",
        "Data Explorer",
        "About RetailSense"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### 📌 Project Information")

st.sidebar.caption("""
**Project:** RetailSense

**Domain:** Data Science

**Language:** Python

**Libraries:** Pandas, NumPy,
Matplotlib, Seaborn and Streamlit

**Approach:** Data Cleaning,
EDA, Statistical Analysis and
Rule-Based Recommendations.
""")


# ============================================================
# 8. DASHBOARD HEADER
# ============================================================

st.title("🛒 RetailSense")

st.markdown("""
### Data Science-Based Retail Product Recommendation
### & Customer Analytics System

Explore retail products, customer behaviour, purchasing
patterns, product performance and category trends through
interactive data analytics.
""")

st.divider()


# ============================================================
# PAGE 1: DASHBOARD OVERVIEW
# ============================================================

if page == "Dashboard Overview":

    st.header("📊 Dashboard Overview")

    st.write(
        "A quick overview of retail products, customers "
        "and shopping activities."
    )

    total_products = len(products)

    total_customers = len(customers)

    total_interactions = len(interactions)

    total_purchases = len(
        interactions[
            interactions["event_type"] == "purchase"
        ]
    )

    estimated_sales = get_estimated_sales(
        interactions,
        products
    )["estimated_sales"].sum()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Products",
        f"{total_products:,}"
    )

    col2.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col3.metric(
        "Total Interactions",
        f"{total_interactions:,}"
    )

    col4.metric(
        "Total Purchases",
        f"{total_purchases:,}"
    )

    col5.metric(
        "Estimated Sales",
        f"₹{estimated_sales:,.2f}"
    )

    st.divider()

    # Category distribution
    st.subheader("Product Category Distribution")

    category_data = products["category"].value_counts()

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(figsize=(7, 5))

        category_data.plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax,
            startangle=90
        )

        ax.set_ylabel("")

        ax.set_title("Category Distribution")

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        category_df = (
            products["category"]
            .value_counts()
            .reset_index()
        )

        category_df.columns = [
            "category",
            "product_count"
        ]

        create_bar_chart(
            category_df,
            "category",
            "product_count",
            "Products by Category"
        )

    st.divider()

    # Interaction distribution
    st.subheader("Customer Interaction Distribution")

    interaction_data = (
        interactions["event_type"]
        .value_counts()
        .reset_index()
    )

    interaction_data.columns = [
        "event_type",
        "count"
    ]

    create_bar_chart(
        interaction_data,
        "event_type",
        "count",
        "Customer Shopping Activities"
    )

    st.info(
        "Estimated sales are calculated using one unit per "
        "purchase event at the listed product price. "
        "They are not verified transaction revenue."
    )


# ============================================================
# PAGE 2: PRODUCT ANALYSIS
# ============================================================

elif page == "Product Analysis":

    st.header("📦 Product Analysis")

    st.write(
        "Analyse product categories, prices, stock levels "
        "and product popularity."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Products",
        len(products)
    )

    col2.metric(
        "Average Price",
        f"₹{products['price'].mean():,.2f}"
    )

    col3.metric(
        "Total Stock",
        f"{products['stock'].sum():,}"
    )

    st.divider()

    # Category filter
    categories = sorted(
        products["category"].dropna().unique()
    )

    selected_category = st.selectbox(
        "Select Product Category",
        ["All Categories"] + categories
    )

    if selected_category == "All Categories":

        filtered_products = products.copy()

    else:

        filtered_products = products[
            products["category"] == selected_category
        ]

    st.subheader("Product Details")

    show_dataframe(filtered_products)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Products by Category")

        category_count = (
            filtered_products["category"]
            .value_counts()
            .reset_index()
        )

        category_count.columns = [
            "category",
            "count"
        ]

        create_bar_chart(
            category_count,
            "category",
            "count",
            "Category Distribution"
        )

    with col2:

        st.subheader("Product Price Distribution")

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.histplot(
            filtered_products["price"].dropna(),
            bins=15,
            kde=True,
            ax=ax
        )

        ax.set_title("Product Price Distribution")

        ax.set_xlabel("Product Price")

        ax.set_ylabel("Frequency")

        st.pyplot(fig)

        plt.close(fig)

    st.divider()

    st.subheader("Top Popular Products")

    popular_products = get_popular_products(
        products,
        interactions,
        top_n=10
    )

    show_dataframe(popular_products)

    download_csv(
        popular_products,
        "popular_products.csv",
        "Download Popular Products"
    )

# ============================================================
# PAGE 3: CUSTOMER ANALYTICS
# ============================================================

elif page == "Customer Analytics":

    st.header("👥 Customer Analytics")

    st.write("""
    Analyse customer demographics, locations, age groups
    and shopping behaviour using interactive filters.
    """)

    st.divider()

    # --------------------------------------------------------
    # CUSTOMER FILTERS
    # --------------------------------------------------------

    st.subheader("🔎 Customer Filters")

    col1, col2 = st.columns(2)

    with col1:

        cities = sorted(
            customers["city"].dropna().unique()
        )

        selected_city = st.selectbox(
            "Select Customer City",
            ["All Cities"] + cities
        )

    with col2:

        min_age = int(customers["age"].min())
        max_age = int(customers["age"].max())

        age_range = st.slider(
            "Select Customer Age Range",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age)
        )

    filtered_customers = customers.copy()

    if selected_city != "All Cities":

        filtered_customers = filtered_customers[
            filtered_customers["city"] == selected_city
        ]

    filtered_customers = filtered_customers[
        filtered_customers["age"].between(
            age_range[0],
            age_range[1]
        )
    ]

    # --------------------------------------------------------
    # CUSTOMER METRICS
    # --------------------------------------------------------

    st.divider()

    st.subheader("📊 Customer Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Filtered Customers",
        len(filtered_customers)
    )

    col2.metric(
        "Average Age",
        f"{filtered_customers['age'].mean():.1f}"
        if not filtered_customers.empty else "N/A"
    )

    col3.metric(
        "Cities Represented",
        filtered_customers["city"].nunique()
    )

    st.divider()

    if filtered_customers.empty:

        st.warning("No customers match the selected filters.")

    else:

        # ----------------------------------------------------
        # AGE DISTRIBUTION
        # ----------------------------------------------------

        st.subheader("📈 Customer Age Distribution")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.histplot(
            data=filtered_customers,
            x="age",
            bins=15,
            kde=True,
            ax=ax
        )

        ax.set_title("Customer Age Distribution")

        ax.set_xlabel("Customer Age")

        ax.set_ylabel("Number of Customers")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.divider()

        # ----------------------------------------------------
        # CITY-WISE CUSTOMER ANALYSIS
        # ----------------------------------------------------

        st.subheader("🏙️ City-wise Customer Analysis")

        city_data = (
            filtered_customers["city"]
            .value_counts()
            .reset_index()
        )

        city_data.columns = [
            "city",
            "customer_count"
        ]

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            data=city_data,
            x="city",
            y="customer_count",
            ax=ax
        )

        ax.set_title("Customers by City")

        ax.set_xlabel("City")

        ax.set_ylabel("Number of Customers")

        plt.xticks(rotation=35, ha="right")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.divider()

        # ----------------------------------------------------
        # CUSTOMER ACTIVITY ANALYSIS
        # ----------------------------------------------------

        st.subheader("🛍️ Shopping Activity Analysis")

        filtered_customer_ids = (
            filtered_customers["customer_id"].unique()
        )

        customer_interactions = interactions[
            interactions["customer_id"].isin(
                filtered_customer_ids
            )
        ].copy()

        activity_data = (
            customer_interactions["event_type"]
            .value_counts()
            .reindex(
                ["view", "cart", "purchase"],
                fill_value=0
            )
            .reset_index()
        )

        activity_data.columns = [
            "activity",
            "count"
        ]

        col1, col2, col3 = st.columns(3)

        activity_counts = dict(
            zip(
                activity_data["activity"],
                activity_data["count"]
            )
        )

        col1.metric(
            "Product Views",
            int(activity_counts["view"])
        )

        col2.metric(
            "Add to Cart",
            int(activity_counts["cart"])
        )

        col3.metric(
            "Purchases",
            int(activity_counts["purchase"])
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        sns.barplot(
            data=activity_data,
            x="activity",
            y="count",
            ax=ax
        )

        ax.set_title("Shopping Activities of Filtered Customers")

        ax.set_xlabel("Activity Type")

        ax.set_ylabel("Number of Activities")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.divider()

        # ----------------------------------------------------
        # CUSTOMER RECORDS
        # ----------------------------------------------------

        st.subheader("📋 Filtered Customer Records")

        st.dataframe(
            filtered_customers,
            use_container_width=True,
            height=350
        )

        download_csv(
            filtered_customers,
            "filtered_customer_report.csv",
            "📥 Download Filtered Customer Report"
        )

        st.divider()

        # ----------------------------------------------------
        # CUSTOMER INTERACTION RECORDS
        # ----------------------------------------------------

        st.subheader("Customer Interaction Records")

        st.dataframe(
            customer_interactions,
            use_container_width=True,
            height=350
        )

        download_csv(
            customer_interactions,
            "filtered_customer_interactions.csv",
            "📥 Download Customer Interaction Report"
        )

# ============================================================
# PAGE 4: PRODUCT RECOMMENDATIONS
# ============================================================

elif page == "Product Recommendations":

    st.header("🎯 Product Recommendations")

    st.write("""
    This section generates product recommendations using
    customer activity, product popularity and category
    preferences.

    The recommendation approach is rule-based and statistical.
    It does not use Machine Learning.
    """)

    customer_ids = customers["customer_id"].dropna().unique()

    selected_customer = st.selectbox(
        "Select Customer ID",
        sorted(customer_ids)
    )

    st.divider()

    # Customer information
    customer_info = customers[
        customers["customer_id"] == selected_customer
    ]

    if not customer_info.empty:

        customer = customer_info.iloc[0]

        st.subheader("Selected Customer Information")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Customer ID",
            str(customer["customer_id"])
        )

        col2.metric(
            "Customer Name",
            str(customer["customer_name"])
        )

        col3.metric(
            "City",
            str(customer["city"])
        )

    st.divider()

    # Category preferences
    st.subheader("Customer Category Preferences")

    preferences = get_customer_preferences(
        selected_customer,
        interactions
    )

    show_dataframe(preferences)

    st.divider()

    # Recommendations
    st.subheader("Recommended Products")

    recommendations = get_customer_recommendations(
        selected_customer,
        products,
        interactions,
        top_n=10
    )

    if recommendations is not None and not recommendations.empty:

        st.success(
            f"{len(recommendations)} product recommendations generated."
        )

        show_dataframe(recommendations)

        download_csv(
            recommendations,
            "customer_recommendations.csv",
            "Download Recommendations"
        )

    else:

        show_empty_message(
            "No recommendations available for this customer."
        )

    st.divider()

    # Frequently bought together
    st.subheader("Frequently Bought Together")

    product_options = products[
        ["product_id", "product_name"]
    ].drop_duplicates()

    selected_product = st.selectbox(
        "Select a Product",
        product_options["product_id"].tolist(),
        format_func=lambda product_id: (
            product_options.loc[
                product_options["product_id"] == product_id,
                "product_name"
            ].iloc[0]
            + f" ({product_id})"
        )
    )

    related_products = get_frequently_bought_together(
        selected_product,
        products,
        interactions,
        top_n=5
    )

    show_dataframe(related_products)


# ============================================================
# PAGE 5: SALES ANALYTICS
# ============================================================

elif page == "Sales Analytics":

    st.header("💰 Sales Analytics")

    st.write(
        "Explore estimated sales performance, purchase trends "
        "and category-wise sales."
    )

    sales_data = get_estimated_sales(
        interactions,
        products
    )

    if sales_data.empty:

        show_empty_message(
            "No purchase records are available."
        )

    else:

        total_sales = sales_data["estimated_sales"].sum()

        total_orders = sales_data["order_id"].nunique()

        average_sales = (
            total_sales / total_orders
            if total_orders > 0 else 0
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Estimated Sales Value",
            f"₹{total_sales:,.2f}"
        )

        col2.metric(
            "Unique Purchase Orders",
            f"{total_orders:,}"
        )

        col3.metric(
            "Average Estimated Value per Order",
            f"₹{average_sales:,.2f}"
        )

        st.divider()

        # Date filtering
        st.subheader("Filter Sales by Date")

        if sales_data["event_date"].notna().any():

            min_date = sales_data["event_date"].min().date()

            max_date = sales_data["event_date"].max().date()

            selected_dates = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            if isinstance(selected_dates, tuple) and len(selected_dates) == 2:

                start_date, end_date = selected_dates

                filtered_sales = sales_data[
                    (sales_data["event_date"].dt.date >= start_date)
                    &
                    (sales_data["event_date"].dt.date <= end_date)
                ]

            else:

                filtered_sales = sales_data.copy()

        else:

            filtered_sales = sales_data.copy()

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Estimated Sales by Category")

            category_sales = (
                filtered_sales
                .groupby("category")["estimated_sales"]
                .sum()
                .reset_index()
                .sort_values(
                    "estimated_sales",
                    ascending=False
                )
            )

            create_bar_chart(
                category_sales,
                "category",
                "estimated_sales",
                "Category-wise Estimated Sales"
            )

        with col2:

            st.subheader("Top Revenue Products")

            product_sales = (
                filtered_sales
                .groupby(
                    ["product_id", "product_name"]
                )["estimated_sales"]
                .sum()
                .reset_index()
                .sort_values(
                    "estimated_sales",
                    ascending=False
                )
                .head(10)
            )

            create_bar_chart(
                product_sales,
                "product_name",
                "estimated_sales",
                "Top 10 Products by Estimated Sales"
            )

        st.divider()

        st.subheader("Monthly Estimated Sales Trend")

        monthly_sales = (
            filtered_sales
            .dropna(subset=["event_date"])
            .set_index("event_date")
            .resample("ME")["estimated_sales"]
            .sum()
            .reset_index()
        )

        if not monthly_sales.empty:

            fig, ax = plt.subplots(figsize=(11, 5))

            sns.lineplot(
                data=monthly_sales,
                x="event_date",
                y="estimated_sales",
                marker="o",
                ax=ax
            )

            ax.set_title("Monthly Estimated Sales")

            ax.set_xlabel("Month")

            ax.set_ylabel("Estimated Sales Value")

            plt.xticks(rotation=35)

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        st.divider()

        st.subheader("Sales Records")

        show_dataframe(filtered_sales)

        download_csv(
            filtered_sales,
            "estimated_sales_report.csv",
            "Download Sales Report"
        )

        st.info(
            "Sales values are estimates based on one unit per "
            "purchase event at the listed product price. "
            "Actual quantities, discounts and transaction amounts "
            "are not available in this dataset."
        )


# ============================================================
# PAGE 6: ADVANCED ANALYTICS & REPORTS
# ============================================================

elif page == "Advanced Analytics & Reports":

    st.header("📈 Advanced Analytics & Reports")

    st.write("""
    This section provides deeper insights into customer
    activity, customer segmentation, product performance
    and category performance.
    """)

    try:

        # Customer activity
        activity_report = customer_activity_analysis(
            customers,
            interactions
        )

        # Customer segmentation
        segmentation_report = customer_segmentation(
            activity_report
        )

        # Product performance
        product_report = product_performance_analysis(
            products,
            interactions
        )

        # Category performance
        category_report = category_performance_analysis(
            products,
            interactions
        )

        # Customer activity section
        st.subheader("1. Customer Activity Analysis")

        if not activity_report.empty:

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Customers with Activity",
                len(activity_report)
            )

            col2.metric(
                "Total Views",
                int(activity_report["view"].sum())
            )

            col3.metric(
                "Total Purchases",
                int(activity_report["purchase"].sum())
            )

            show_dataframe(activity_report)

            download_csv(
                activity_report,
                "customer_activity_report.csv",
                "Download Customer Activity Report"
            )

        else:

            show_empty_message()

        st.divider()

        # Segmentation
        st.subheader("2. Customer Segmentation")

        if (
            segmentation_report is not None
            and not segmentation_report.empty
            and "customer_segment" in segmentation_report.columns
        ):

            segment_counts = (
                segmentation_report["customer_segment"]
                .value_counts()
                .reset_index()
            )

            segment_counts.columns = [
                "segment",
                "customer_count"
            ]

            col1, col2 = st.columns(2)

            with col1:

                fig, ax = plt.subplots(figsize=(7, 5))

                ax.pie(
                    segment_counts["customer_count"],
                    labels=segment_counts["segment"],
                    autopct="%1.1f%%",
                    startangle=90
                )

                ax.set_title("Customer Segment Distribution")

                st.pyplot(fig)

                plt.close(fig)

            with col2:

                create_bar_chart(
                    segment_counts,
                    "segment",
                    "customer_count",
                    "Customers by Segment"
                )

            show_dataframe(segmentation_report)

            download_csv(
                segmentation_report,
                "customer_segmentation_report.csv",
                "Download Segmentation Report"
            )

        else:

            show_empty_message(
                "Customer segmentation report is unavailable."
            )

        st.divider()

        # Product performance
        st.subheader("3. Product Performance Analysis")

        if not product_report.empty:

            show_dataframe(product_report)

            download_csv(
                product_report,
                "product_performance_report.csv",
                "Download Product Performance Report"
            )

        else:

            show_empty_message()

        st.divider()

        # Category performance
        st.subheader("4. Category Performance Analysis")

        if not category_report.empty:

            show_dataframe(category_report)

            download_csv(
                category_report,
                "category_performance_report.csv",
                "Download Category Performance Report"
            )

        else:

            show_empty_message()

    except Exception as error:

        st.error(
            "An error occurred while generating advanced analytics."
        )

        st.exception(error)

        st.info(
            "Run python -m modules.advanced_analytics "
            "and check the output before reopening the dashboard."
        )


# ============================================================
# PAGE 7: DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.header("🔍 Data Explorer")

    st.write(
        "Explore the cleaned datasets used by the RetailSense application."
    )

    dataset_name = st.selectbox(
        "Select Dataset",
        [
            "Products",
            "Customers",
            "Interactions"
        ]
    )

    if dataset_name == "Products":

        selected_data = products.copy()

    elif dataset_name == "Customers":

        selected_data = customers.copy()

    else:

        selected_data = interactions.copy()

    st.subheader(f"{dataset_name} Dataset")

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Rows",
        len(selected_data)
    )

    col2.metric(
        "Total Columns",
        len(selected_data.columns)
    )

    st.divider()

    # Search
    search_text = st.text_input(
        "Search Dataset",
        placeholder="Enter a value to search..."
    )

    if search_text:

        mask = selected_data.astype(str).apply(
            lambda column: column.str.contains(
                search_text,
                case=False,
                na=False
            )
        ).any(axis=1)

        selected_data = selected_data[mask]

    st.subheader("Dataset Records")

    show_dataframe(selected_data, height=450)

    st.divider()

    st.subheader("Dataset Information")

    info_data = pd.DataFrame({
        "Column Name": selected_data.columns,
        "Data Type": selected_data.dtypes.astype(str).values,
        "Missing Values": selected_data.isnull().sum().values,
        "Unique Values": selected_data.nunique().values
    })

    show_dataframe(info_data)

    download_csv(
        selected_data,
        f"{dataset_name.lower()}_dataset.csv",
        "Download Selected Dataset"
    )


# ============================================================
# PAGE 8: ABOUT RETAILSENSE
# ============================================================

elif page == "About RetailSense":

    st.header("📘 About RetailSense")

    st.write("""
    **RetailSense** is a Data Science-based retail product
    recommendation and customer analytics application.

    It demonstrates how structured retail datasets can be
    processed and analysed to understand customer activities,
    product performance, purchasing patterns and category trends.

    The application uses synthetic academic datasets.
    """)

    st.divider()

    st.subheader("Project Objectives")

    objectives = [
        "Generate and analyse structured retail datasets.",
        "Clean and prepare raw datasets for analysis.",
        "Understand customer shopping behaviour.",
        "Identify popular products and categories.",
        "Generate rule-based product recommendations.",
        "Analyse estimated sales performance.",
        "Perform customer segmentation using predefined rules.",
        "Present analytical results through an interactive dashboard."
    ]

    for objective in objectives:

        st.markdown(f"- {objective}")

    st.divider()

    st.subheader("Technologies Used")

    technologies = {
        "Programming Language": "Python",
        "Data Analysis": "Pandas, NumPy",
        "Data Visualization": "Matplotlib, Seaborn",
        "Dashboard": "Streamlit",
        "Data Storage": "CSV Datasets",
        "Recommendation Approach": "Rule-Based Statistical Analysis"
    }

    tech_df = pd.DataFrame(
        list(technologies.items()),
        columns=["Technology Area", "Tools"]
    )

    show_dataframe(tech_df)

    st.divider()

    st.subheader("Project Methodology")

    methodology = [
        "Data Generation",
        "Data Cleaning and Preprocessing",
        "Exploratory Data Analysis",
        "Customer Behaviour Analysis",
        "Product Recommendation",
        "Advanced Analytics",
        "Dashboard Development"
    ]

    for index, step in enumerate(methodology, start=1):

        st.markdown(f"**{index}.** {step}")

    st.divider()

    st.subheader("Important Project Information")

    st.info("""
    RetailSense is an academic Data Science demonstration
    developed using synthetic retail data.

    It is not connected to Amazon or any live retail platform.

    Product recommendations and customer segments are generated
    using predefined statistical and business rules, not Machine
    Learning.
    """)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray;">

        <p><b>RetailSense</b></p>

        <p>
        Data Science-Based Retail Product Recommendation
        & Customer Analytics System
        </p>

        <p>
        Developed using Python, Pandas, NumPy,
        Matplotlib, Seaborn and Streamlit
        </p>

        <p>Academic Project | 2026</p>

    </div>
    """,
    unsafe_allow_html=True
)