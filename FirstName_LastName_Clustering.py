# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

# Load the datasets
customers = pd.read_csv(r"C:\Users\91739\Downloads\Customers.csv")

products = pd.read_csv(r"C:\Users\91739\Downloads\Products.csv")
    
transactions = pd.read_csv(r"C:\Users\91739\Downloads\Transactions.csv")


# Convert date columns to datetime format
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

# Merge datasets for further analysis
merged_data = transactions.merge(customers, on='CustomerID').merge(products, on='ProductID')

# Customer Segmentation / Clustering
# Feature engineering for clustering
clustering_features = merged_data.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'TotalValue': 'sum',
    'Price': 'mean'}).reset_index()

      
# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clustering_features['Cluster'] = kmeans.fit_predict(clustering_features[['Quantity', 'TotalValue', 'Price']])

# Calculate Davies-Bouldin Index
db_index = davies_bouldin_score(clustering_features[['Quantity', 'TotalValue', 'Price']], clustering_features['Cluster'])

# Visualize clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=clustering_features, x='TotalValue', y='Quantity', hue='Cluster', palette='viridis', s=100)
plt.title('Customer Clusters')
plt.xlabel('Total Transaction Value')
plt.ylabel('Total Quantity Purchased')
plt.legend(title='Cluster')
plt.show()

# Display clustering results
cluster_summary = clustering_features.groupby('Cluster').mean()

# Output results
print(f"Davies-Bouldin Index: {db_index}")
print("Cluster Summary:\n", cluster_summary)


