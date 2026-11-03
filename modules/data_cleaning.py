import os
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATASETS
# --------------------------------------------------

def load_datasets():

    products = pd.read_csv(
        os.path.join(DATA_DIR, "products.csv")
    )

    customers = pd.read_csv(
        os.path.join(DATA_DIR, "customers.csv")
    )

    interactions = pd.read_csv(
        os.path.join(DATA_DIR, "interactions.csv")
    )

    return products, customers, interactions


# --------------------------------------------------
# 3. DISPLAY DATASET INFORMATION
# --------------------------------------------------

def dataset_information(name, dataframe):

    print("\n-----------------------------------")
    print(name)
    print("-----------------------------------")

    print("Dataset Shape:", dataframe.shape)

    print("\nColumns:")
    print(dataframe.columns.tolist())

    print("\nDuplicate Rows:", dataframe.duplicated().sum())

    print("\nMissing Values:")
    print(dataframe.isnull().sum())


# --------------------------------------------------
# 4. CLEAN PRODUCTS DATASET
# --------------------------------------------------

def clean_products(products):

    products = products.copy()

    # Remove duplicate product IDs.
    products = products.drop_duplicates(
        subset=["product_id"]
    )

    # Remove rows with missing essential values.
    products = products.dropna(
        subset=[
            "product_id",
            "product_name",
            "category",
            "price"
        ]
    )

    # Clean text columns.
    for column in ["product_name", "category", "brand"]:
        products[column] = (
            products[column]
            .astype("string")
            .str.strip()
        )

    # Convert numeric columns.
    products["price"] = pd.to_numeric(
        products["price"],
        errors="coerce"
    )

    products["stock"] = pd.to_numeric(
        products["stock"],
        errors="coerce"
    )

    # Remove invalid prices.
    products = products[
        (products["price"] > 0)
        & (products["price"] != float("inf"))
    ]

    # Remove invalid stock values.
    products = products[
        (products["stock"] >= 0)
        & (products["stock"] % 1 == 0)
    ]

    products["stock"] = products["stock"].astype(int)

    products = products.reset_index(drop=True)

    return products


# --------------------------------------------------
# 5. CLEAN CUSTOMERS DATASET
# --------------------------------------------------

def clean_customers(customers):

    customers = customers.copy()

    # Remove duplicate customer IDs.
    customers = customers.drop_duplicates(
        subset=["customer_id"]
    )

    # Remove missing essential values.
    customers = customers.dropna(
        subset=[
            "customer_id",
            "customer_name",
            "city",
            "age",
            "registration_date"
        ]
    )

    # Clean text columns.
    for column in ["customer_name", "city"]:
        customers[column] = (
            customers[column]
            .astype("string")
            .str.strip()
        )

    # Convert age into numeric format.
    customers["age"] = pd.to_numeric(
        customers["age"],
        errors="coerce"
    )

    customers = customers[
        (customers["age"] >= 18)
        & (customers["age"] <= 100)
    ]

    customers["age"] = customers["age"].astype(int)

    # Convert registration date.
    customers["registration_date"] = pd.to_datetime(
        customers["registration_date"],
        errors="coerce"
    )

    customers = customers.dropna(
        subset=["registration_date"]
    )

    customers = customers.reset_index(drop=True)

    return customers


# --------------------------------------------------
# 6. CLEAN INTERACTIONS DATASET
# --------------------------------------------------

def clean_interactions(interactions, customers, products):

    interactions = interactions.copy()

    # Remove duplicate event IDs.
    interactions = interactions.drop_duplicates(
        subset=["event_id"]
    )

    # Remove rows with missing essential values.
    interactions = interactions.dropna(
        subset=[
            "event_id",
            "customer_id",
            "product_id",
            "event_type",
            "event_date"
        ]
    )

    # Clean event type.
    interactions["event_type"] = (
        interactions["event_type"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Keep only valid event types.
    valid_events = ["view", "cart", "purchase"]

    interactions = interactions[
        interactions["event_type"].isin(valid_events)
    ]

    # Convert event date.
    interactions["event_date"] = pd.to_datetime(
        interactions["event_date"],
        errors="coerce"
    )

    interactions = interactions.dropna(
        subset=["event_date"]
    )

    # Validate customer IDs.
    valid_customer_ids = set(
        customers["customer_id"]
    )

    interactions = interactions[
        interactions["customer_id"].isin(valid_customer_ids)
    ]

    # Validate product IDs.
    valid_product_ids = set(
        products["product_id"]
    )

    interactions = interactions[
        interactions["product_id"].isin(valid_product_ids)
    ]

    # Purchase events must have an order ID.
    purchase_mask = (
        interactions["event_type"] == "purchase"
    )

    interactions = interactions[
        (~purchase_mask)
        | interactions["order_id"].notna()
    ]

    # Non-purchase events do not require an order ID.
    interactions.loc[
        interactions["event_type"] != "purchase",
        "order_id"
    ] = pd.NA

    # --------------------------------------------------
    # ADD PRODUCT CATEGORY TO INTERACTIONS
    # --------------------------------------------------

    interactions = interactions.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left"
    )

    interactions = interactions.rename(
        columns={
            "category": "product_category"
        }
    )

    # Reset index.
    interactions = interactions.reset_index(drop=True)

    return interactions


# --------------------------------------------------
# 7. SAVE CLEANED DATASETS
# --------------------------------------------------

def save_cleaned_data(products, customers, interactions):

    products.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "cleaned_products.csv"
        ),
        index=False
    )

    customers.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "cleaned_customers.csv"
        ),
        index=False
    )

    interactions.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "cleaned_interactions.csv"
        ),
        index=False
    )

    print("\nAll cleaned datasets saved successfully!")


# --------------------------------------------------
# 8. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("\n===================================")
    print(" RETAILSENSE DATA CLEANING SYSTEM ")
    print("===================================")

    # Load original datasets.
    products, customers, interactions = load_datasets()

    print("\nOriginal datasets loaded successfully!")

    # Display original dataset information.
    dataset_information("PRODUCTS DATASET", products)

    dataset_information("CUSTOMERS DATASET", customers)

    dataset_information("INTERACTIONS DATASET", interactions)

    # Clean products.
    print("\nCleaning products dataset...")

    products = clean_products(products)

    # Clean customers.
    print("Cleaning customers dataset...")

    customers = clean_customers(customers)

    # Clean interactions and add product category.
    print("Cleaning interactions dataset...")

    interactions = clean_interactions(
        interactions,
        customers,
        products
    )

    # Display cleaned dataset information.
    print("\n===================================")
    print(" CLEANED DATASET INFORMATION ")
    print("===================================")

    dataset_information("CLEANED PRODUCTS", products)

    dataset_information("CLEANED CUSTOMERS", customers)

    dataset_information("CLEANED INTERACTIONS", interactions)

    # Save cleaned datasets.
    save_cleaned_data(
        products,
        customers,
        interactions
    )

    print("\nInteraction columns:")
    print(interactions.columns.tolist())

    print("\n===================================")
    print(" DATA CLEANING COMPLETED SUCCESSFULLY ")
    print("===================================")


if __name__ == "__main__":
    main()