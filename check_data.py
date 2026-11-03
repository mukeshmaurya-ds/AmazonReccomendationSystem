import pandas as pd

products = pd.read_csv("data/products.csv")

customers = pd.read_csv("data/customers.csv")

interactions = pd.read_csv("data/interactions.csv")


print("\nPRODUCT DATASET")

print(products.head())

print("\nCUSTOMER DATASET")

print(customers.head())

print("\nINTERACTION DATASET")

print(interactions.head())


print("\nDATASET SHAPES")

print("Products:", products.shape)

print("Customers:", customers.shape)

print("Interactions:", interactions.shape)


print("\nMISSING VALUES")

print(interactions.isnull().sum())


print("\nDATASET VERIFICATION COMPLETED")