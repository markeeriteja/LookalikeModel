
# Customer Segmentation and Lookalike Analysis

## Overview
This project focuses on analyzing customer, product, and transaction data to implement clustering, similarity-based recommendation systems, and exploratory data analysis. The key goals are to:

1. Segment customers based on transaction behavior.
2. Identify lookalike customers using cosine similarity.
3. Evaluate clustering performance using the Davies-Bouldin Index (DBI).
4. Visualize customer insights for actionable recommendations.

---

## Key Components

### 1. **Dataset Description**

- **`Customers.csv`**: Contains customer information.
  - `CustomerID`: Unique identifier for each customer.
  - `CustomerName`: Name of the customer.
  - `Region`: Geographical region of the customer.
  - `SignupDate`: Date of account creation.

- **`Products.csv`**: Contains product information.
  - `ProductID`: Unique identifier for each product.
  - `ProductName`: Name of the product.
  - `Category`: Product category.
  - `Price`: Price of the product.

- **`Transactions.csv`**: Contains transaction details.
  - `TransactionID`: Unique identifier for each transaction.
  - `CustomerID`: Associated customer ID.
  - `ProductID`: Associated product ID.
  - `TransactionDate`: Date of the transaction.
  - `Quantity`: Quantity purchased.
  - `TotalValue`: Total value of the transaction.

---

### 2. **Analysis and Methods**

#### a. **Exploratory Data Analysis (EDA)**

- Merged datasets to create a unified view of customer, product, and transaction data.
- Aggregated key statistics like:
  - Number of unique customers, products, and regions.
  - Top-performing products based on purchase count.
  
Visualization of insights:
- **Customer Distribution by Region**: Visualized using Seaborn bar plots.
- **Top 5 Products by Purchases**: Highlighted using bar plots to identify popular products.

#### b. **Lookalike Customer Recommendation System**

- **Feature Aggregation**:
  - `total_spent`: Total spending by each customer.
  - `transaction_count`: Number of transactions.
  - `distinct_products`: Number of unique products purchased.

- **Data Normalization**:
  - Used `MinMaxScaler` to normalize features between 0 and 1.

- **Cosine Similarity**:
  - Computed pairwise similarity between customers using normalized features.
  - Generated a dictionary of lookalike results for the given 200 customers, with the top 3 most similar customers for each.

#### c. **Customer Segmentation with K-Means Clustering**

- Applied **K-Means Clustering** to segment customers into 4 clusters based on:
  - Spending patterns.
  - Transaction frequency.
  - Product diversity.

- Evaluated clustering performance using **Davies-Bouldin Index (DBI)**. Lower DBI values indicate better clustering.

---

### 3. **Code Snippets and Results**

#### a. **Lookalike Recommendation**
```python
lookalike_results = {}
for customer_id in similarity_df.indexs:
    similar_customers = similarity_df.loc[customer_id].sort_values(ascending=False)[1:4]
    lookalike_results[customer_id] = list(zip(similar_customers.index, similar_customers.values))

# Example: Lookalikes for customer 'C0001'
lookalike_results['C0001']
```

**Output**:
A dictionary with customer IDs as keys and their top 3 similar customers with similarity scores.

#### b. **K-Means Clustering**
```python
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(customer_features_scaled)
db_index = davies_bouldin_score(customer_features_scaled, clusters)
customer_features["Cluster"] = clusters
```

**Output**:
- `customer_features`: DataFrame with cluster assignments for each customer.
- `db_index`: Davies-Bouldin Index score.

---

## Insights and Recommendations

### 1. **Customer Segmentation**
- Clusters provide distinct customer profiles:
  - High-spending frequent buyers.
  - Low-spending occasional buyers.
  - Diverse product enthusiasts.

### 2. **Targeted Marketing**
- Use clusters to design specific campaigns:
  - Upselling to high spenders.
  - Engaging low-spending customers with discounts.

### 3. **Personalized Recommendations**
- Lookalike results help identify customers with similar purchase behaviors, enabling personalized product recommendations.

---



## Requirements

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

---

## Running the Code

1. Ensure all datasets (`Customers.csv`, `Products.csv`, `Transactions.csv`) are in the working directory.
2. Install required libraries using:
   ```bash
   pip install pandas scikit-learn numpy matplotlib seaborn
   ```
3. Run the Python script step by step.

---


