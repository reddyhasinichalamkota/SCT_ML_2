# SCT_ML_2
# Customer Segmentation using K-Means Clustering

## Overview
This project applies the **K-Means Clustering** algorithm to segment customers of a retail store based on their purchase behavior. Customer segmentation helps businesses identify different groups of customers and enables data-driven marketing strategies, personalized recommendations, and improved customer retention.

## Objective
The objective of this project is to analyze customer purchasing patterns and group customers into distinct clusters based on similar characteristics such as income and spending habits.

## Key Features
- Data preprocessing and cleaning  
- Exploratory Data Analysis (EDA)  
- Optimal cluster selection using the Elbow Method  
- Customer segmentation using K-Means Clustering  
- Cluster visualization and interpretation  
- Business insights generation  

## Technologies Used
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

## Dataset Information
The dataset includes customer information such as:

- Customer ID  
- Gender  
- Age  
- Annual Income  
- Spending Score  

## Methodology

### 1. Data Collection
Loaded the customer dataset and reviewed its structure.

### 2. Data Preprocessing
Handled missing values, selected relevant features, and prepared the data for clustering.

### 3. Finding Optimal Clusters
Used the **Elbow Method** to determine the ideal number of clusters for the dataset.

### 4. Model Training
Applied the K-Means algorithm to group customers into clusters.

### 5. Visualization
Generated scatter plots to visualize customer segments.

## Output
The model segments customers into categories such as:

- High Income, High Spending  
- High Income, Low Spending  
- Low Income, High Spending  
- Low Income, Low Spending  
- Average Customers  

## Project Structure

```bash
SCT_ML_2/
│── customer_segmentation.py
│── Mall_Customers.csv
│── README.md
│── requirements.txt
│── output.png
