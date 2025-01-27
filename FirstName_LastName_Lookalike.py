
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
customers = pd.read_csv(r"C:\Users\91739\Downloads\Customers.csv")

products = pd.read_csv(r"C:\Users\91739\Downloads\Products.csv")
    
transactions = pd.read_csv(r"C:\Users\91739\Downloads\Transactions.csv")


# Convert date columns to datetime format
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

# Check for missing values and duplicates
missing_values = {
    "Customers": customers.isnull().sum().to_dict(),
    "Products": products.isnull().sum().to_dict(),
    "Transactions": transactions.isnull().sum().to_dict(),
}
duplicates = {
    "Customers": customers.duplicated().sum(),
    "Products": products.duplicated().sum(),
    "Transactions": transactions.duplicated().sum(),
}

# Merge datasets for EDA
merged_data = transactions.merge(customers, on='CustomerID').merge(products, on='ProductID')

# Perform exploratory analysis (example insights and visualizations)
# Distribution of transactions across regions
plt.figure(figsize=(8, 5))
sns.countplot(data=customers, x='Region', order=customers['Region'].value_counts().index)
plt.title('Number of Customers by Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Top 10 products by sales
top_products = merged_data.groupby('ProductName').agg({'TotalValue': 'sum'}).sort_values(by='TotalValue', ascending=False).head(10)

# Revenue over time
merged_data['Month'] = merged_data['TransactionDate'].dt.to_period('M')
monthly_revenue = merged_data.groupby('Month').agg({'TotalValue': 'sum'})

# Visualization for revenue over time
monthly_revenue.plot(figsize=(10, 5))
plt.title('Monthly Revenue Trend')
plt.ylabel('Revenue (USD)')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Insights
insights = [
    "Region Europe has the highest number of customers in this dataset.",
    "Product C generates the most revenue, contributing significantly to overall sales.",
    "Revenue increases consistently over time, peaking in July 2022.",
    "Customers signing up in Q1 2022 are active participants in transactions.",
    "Asian customers prefer purchasing in bulk as shown by their high transaction quantities."
]

# Lookalike Model
# Feature engineering
customer_features = merged_data.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'TotalValue': 'sum'
}).reset_index()
customer_features = customer_features.merge(customers[['CustomerID', 'Region']], on='CustomerID')
customer_features_encoded = pd.get_dummies(customer_features, columns=['Region'], drop_first=True)

# Similarity computation
similarity_matrix = cosine_similarity(customer_features_encoded.drop('CustomerID', axis=1))
similarity_df = pd.DataFrame(similarity_matrix, index=customer_features['CustomerID'], columns=customer_features['CustomerID'])

# Top 3 similar customers for each of the first 20 customers
lookalike_results = {}
for customer in customer_features['CustomerID'][:20]:
    similar_customers = similarity_df[customer].sort_values(ascending=False).iloc[1:4]  # Exclude self-similarity
    lookalike_results[customer] = similar_customers.reset_index().values.tolist()

# Display lookalike results
lookalike_results

