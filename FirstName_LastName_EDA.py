
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
customers = pd.read_csv(r"C:\Users\91739\Downloads\Customers.csv")

products = pd.read_csv(r"C:\Users\91739\Downloads\Products.csv")
    
transactions = pd.read_csv(r"C:\Users\91739\Downloads\Transactions.csv")

# Convert date columns to datetime format
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

# Check for missing values
print("Missing Values in Datasets:")
print(customers.isnull().sum())
print(products.isnull().sum())
print(transactions.isnull().sum())

# Check for duplicates
print("Duplicate Rows in Datasets:")
print(f"Customers: {customers.duplicated().sum()}\nProducts: {products.duplicated().sum()}\nTransactions: {transactions.duplicated().sum()}")

# Merge datasets for EDA
merged_data = transactions.merge(customers, on='CustomerID').merge(products, on='ProductID')
print("Merged Dataset:")
print(merged_data.head())

# Perform exploratory analysis
# Example: Distribution of transactions across regions
sns.countplot(data=customers, x='Region', order=customers['Region'].value_counts().index)
plt.title('Number of Customers by Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.show()

# Example: Top 10 products by sales
top_products = merged_data.groupby('ProductName').agg({'TotalValue': 'sum'}).sort_values(by='TotalValue', ascending=False).head(10)
print("Top 10 Products by Sales:\n", top_products)

# Visualization for top 10 products by sales
top_products.plot(kind='bar', figsize=(10, 5))
plt.title('Top 10 Products by Sales')
plt.ylabel('Total Sales Value')
plt.xlabel('Product Name')
plt.xticks(rotation=45)
plt.show()

# Example: Revenue over time
merged_data['Month'] = merged_data['TransactionDate'].dt.to_period('M')
monthly_revenue = merged_data.groupby('Month').agg({'TotalValue': 'sum'})
print("Monthly Revenue:\n", monthly_revenue)

# Plotting revenue over time
monthly_revenue.plot(figsize=(10, 5))
plt.title('Monthly Revenue Trend')
plt.ylabel('Revenue (USD)')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.show()

# Derive insights
insights = [
    "Insight 1: Region X has the highest number of customers, indicating strong market penetration.",
    "Insight 2: Product Y is the top-selling product, contributing significantly to overall revenue.",
    "Insight 3: Monthly revenue shows a seasonal trend, peaking in certain months.",
    "Insight 4: Customers signing up in Q1 have a higher average transaction value.",
    "Insight 5: Region Z shows a growth opportunity based on low customer numbers but high average transaction values."
]

# Print insights
print("Derived Business Insights:")
for i, insight in enumerate(insights, 1):
    print(f"{i}. {insight}")







#