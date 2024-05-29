# Customer-Segmentation-Project
Customer Segmentation using K-mean Cluster Algorithm 

Overview
This project implements a customer segmentation model using the K-Means clustering algorithm on an online retail dataset. Customer segmentation is a powerful marketing tool that divides a customer base into groups of individuals with similar characteristics. This enables businesses to tailor their marketing strategies, enhance customer satisfaction, and improve overall business performance.

## Dataset
The dataset used in this project contains transaction data from an online retail store. The key columns in the dataset are:

- InvoiceNo: Unique identifier for each transaction
- StockCode: Unique identifier for each product
- Description: Product description
- Quantity: Number of units purchased
- InvoiceDate: Date and time of the transaction
- UnitPrice: Price per unit of the product
- CustomerID: Unique identifier for each customer
- Country: Country where the customer resides

## Objectives
1. Preprocess the data to handle missing values, duplicate entries, and data transformations.
2. Perform Exploratory Data Analysis (EDA) to gain insights into customer behavior.
3. Implement K-Means clustering to segment customers based on their purchasing patterns.
4. Visualize the clustering results and interpret the customer segments.

## Preprocessing
1. Handling Missing Values: Remove rows with missing CustomerID as they are essential for customer segmentation.
2. Removing Duplicates: Eliminate duplicate entries to ensure data quality.
3. Feature Engineering: Create new features such as TotalSpent (Quantity * UnitPrice) and aggregate data at the customer level.

## Exploratory Data Analysis (EDA)
Several visualizations were created to understand the data better:

- Top Selling Products: Bar chart showing the most frequently purchased products.
- Sales by Country: Bar chart illustrating total sales per country.
- Number of Orders per Customer: Histogram showing the distribution of order counts among customers.
- Distribution of Order Value: Histogram or density plot showing the distribution of total order values.
- Quantity Ordered by Customer: Histogram showing the distribution of quantities purchased by customers.
- Average Order Value by Country: Bar chart displaying the average order value for each country.

## K-Means Clustering
1. Feature Selection: Selected features relevant to customer segmentation, such as TotalSpent, Quantity, and OrderFrequency.
2. Scaling: Standardized the features to ensure fair clustering.
3. Optimal Number of Clusters: Determined the optimal number of clusters using the Elbow Method and Silhouette Score.
4. Model Implementation: Applied the K-Means algorithm to segment the customers.
5. Cluster Analysis: Analyzed the characteristics of each cluster to understand customer segments.
