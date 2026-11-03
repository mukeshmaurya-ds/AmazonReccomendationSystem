import pandas as pd
import numpy as np
import random
import os

from datetime import datetime, timedelta


# ==========================================
# 1. INITIAL CONFIGURATION
# ==========================================

random.seed(42)
np.random.seed(42)

os.makedirs("data", exist_ok=True)

print("=" * 50)
print("       RETAILSENSE DATA GENERATOR")
print("=" * 50)


# ==========================================
# 2. CUSTOMER DATA
# ==========================================

first_names = [
    "Rahul", "Amit", "Priya", "Neha", "Rohit",
    "Sneha", "Ankit", "Pooja", "Karan", "Anjali",
    "Vikas", "Riya", "Arjun", "Simran", "Aditya"
]

cities = [
    "Pune", "Mumbai", "Delhi", "Bangalore",
    "Hyderabad", "Lucknow", "Jaipur",
    "Chennai", "Kolkata", "Nagpur"
]

customers = []

for i in range(1, 251):

    customers.append({

        "customer_id": f"C{i:04d}",

        "customer_name": random.choice(first_names),

        "city": random.choice(cities),

        "age": random.randint(18, 60),

        "registration_date": (
            datetime(2025, 1, 1)
            + timedelta(days=random.randint(0, 300))
        ).strftime("%Y-%m-%d")

    })


customers_df = pd.DataFrame(customers)

customers_df.to_csv(
    "data/customers.csv",
    index=False
)

print("\nCustomer dataset created.")


# ==========================================
# 3. PRODUCT DATA
# ==========================================

product_catalog = {

    "Electronics": [
        "Laptop", "Smartphone", "Tablet",
        "Monitor", "Keyboard", "Wireless Mouse",
        "Headphones", "Speaker", "Webcam"
    ],

    "Accessories": [
        "Laptop Bag", "Mobile Cover",
        "USB Cable", "Power Bank",
        "Laptop Stand", "Charger"
    ],

    "Fashion": [
        "T-Shirt", "Jeans", "Shoes",
        "Jacket", "Watch", "Sunglasses"
    ],

    "Home Appliances": [
        "Mixer Grinder", "Microwave",
        "Refrigerator", "Washing Machine",
        "Electric Kettle", "Vacuum Cleaner"
    ],

    "Books": [
        "Python Programming",
        "Data Science Handbook",
        "Business Analytics",
        "Mathematics Guide",
        "Computer Fundamentals"
    ],

    "Sports": [
        "Cricket Bat", "Football",
        "Badminton Racket", "Yoga Mat",
        "Skipping Rope"
    ]

}

products = []

product_counter = 1

for category, product_names in product_catalog.items():

    for i in range(25):

        product_name = random.choice(product_names)

        price = random.randint(200, 50000)

        products.append({

            "product_id": f"P{product_counter:04d}",

            "product_name": product_name,

            "category": category,

            "price": price,

            "stock": random.randint(0, 100),

            "brand": random.choice([
                "Samsung", "Apple", "HP",
                "Dell", "Lenovo", "Sony",
                "Boat", "Nike", "Generic"
            ])

        })

        product_counter += 1


products_df = pd.DataFrame(products)

products_df.to_csv(
    "data/products.csv",
    index=False
)

print("Product dataset created.")


# ==========================================
# 4. CUSTOMER INTERACTION DATA
# ==========================================

interactions = []

customer_ids = customers_df["customer_id"].tolist()

product_ids = products_df["product_id"].tolist()

start_date = datetime(2025, 11, 1)

event_counter = 1

order_counter = 1


# ------------------------------------------
# A. Generate product views and cart events
# ------------------------------------------

for i in range(1700):

    customer_id = random.choice(customer_ids)

    product_id = random.choice(product_ids)

    event_type = random.choices(
        ["view", "cart"],
        weights=[75, 25]
    )[0]

    event_date = start_date + timedelta(
        days=random.randint(0, 365)
    )

    interactions.append({

        "event_id": f"E{event_counter:06d}",

        "order_id": "",

        "customer_id": customer_id,

        "product_id": product_id,

        "event_type": event_type,

        "event_date": event_date.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    })

    event_counter += 1


# ------------------------------------------
# B. Generate purchase transactions
# ------------------------------------------

for i in range(150):

    order_id = f"O{order_counter:05d}"

    customer_id = random.choice(customer_ids)

    purchase_date = start_date + timedelta(
        days=random.randint(0, 365)
    )

    number_of_products = random.randint(1, 4)

    selected_products = random.sample(
        product_ids,
        number_of_products
    )

    for product_id in selected_products:

        interactions.append({

            "event_id": f"E{event_counter:06d}",

            "order_id": order_id,

            "customer_id": customer_id,

            "product_id": product_id,

            "event_type": "purchase",

            "event_date": purchase_date.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        })

        event_counter += 1

    order_counter += 1


# ------------------------------------------
# C. Save interaction dataset
# ------------------------------------------

interactions_df = pd.DataFrame(interactions)

interactions_df.to_csv(
    "data/interactions.csv",
    index=False
)

print("Interaction dataset created.")


# ==========================================
# 5. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 50)

print("       DATA GENERATION COMPLETED")

print("=" * 50)

print(f"Total Products: {len(products_df)}")

print(f"Total Customers: {len(customers_df)}")

print(f"Total Interactions: {len(interactions_df)}")

print(
    "Total Orders:",
    interactions_df[
        interactions_df["event_type"] == "purchase"
    ]["order_id"].nunique()
)

print("\nEvent Distribution:")

print(
    interactions_df["event_type"].value_counts()
)

print("\nFiles saved successfully inside data/ folder.")

print("=" * 50)