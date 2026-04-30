# SCT_ML_2
# Mall Customer Segmentation using K-Means Clustering

This project applies the **K-Means Clustering** algorithm to segment mall customers based on their purchasing behavior. The model groups customers using important features such as **Age, Annual Income, and Spending Score**.

Customer segmentation helps businesses understand customer behavior, improve marketing strategies, personalize offers, and make data-driven business decisions.

---

## Project Overview

Customer segmentation is an important application of Unsupervised Machine Learning. In this project:

- Customer dataset is loaded and analyzed
- Important numerical features are selected
- Data is preprocessed for clustering
- K-Means algorithm is applied
- Customers are grouped into clusters
- Cluster profiles are analyzed
- Data visualizations are generated
- Business insights are extracted

---

## Objective

The objective of this project is to analyze customer data and divide customers into meaningful groups based on similar characteristics and spending habits.

---

## Key Features

- Data Preprocessing and Cleaning
- Exploratory Data Analysis
- Customer Segmentation using K-Means Clustering
- Cluster Profiling
- Data Visualization
- Business Insight Generation

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Dataset Information

The dataset contains customer details such as:

- Customer ID
- Age
- Annual Income
- Spending Score

---

## Project Structure

```bash
SCT_ML_2/
│── Mall_Customers.csv
│── Mall_customer.py
│── README.md
│── cluster_visualization.png
│── customer_segments.png
│── elbow_method.png
```

---

## Machine Learning Workflow

1. Load customer dataset  
2. Explore dataset statistics  
3. Select important numerical features  
4. Normalize / preprocess data  
5. Apply K-Means Clustering  
6. Assign cluster labels  
7. Analyze cluster averages  
8. Visualize customer segments  
9. Generate business insights  

---

## Model Used

### K-Means Clustering

K-Means is an Unsupervised Machine Learning algorithm used to divide data into groups (clusters) based on similarity.

It works by:

- Selecting K cluster centers
- Assigning points to nearest cluster
- Updating centroids repeatedly
- Creating optimized clusters

---

## Dataset Summary

```text
Total Customers : 200
Average Age     : 38.85
Average Income  : 60.56
Average Score   : 50.20
Minimum Age     : 18
Maximum Age     : 70
```

---

## Cluster Insights

Example customer groups may include:

- High Income, High Spending Customers
- High Income, Low Spending Customers
- Low Income, High Spending Customers
- Young Moderate Spenders
- Budget Conscious Customers

---

## How to Run

### Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run Project

```bash
python Mall_customer.py
```

---

## Sample Output

```text
Mall Customer Segmentation using K-Means

Total Customers : 200

Clustering Completed Successfully
Customer Groups Generated
Visualization Saved
```

---

## Why K-Means?

K-Means Clustering is useful because:

- Easy to understand and implement
- Fast on medium datasets
- Helps discover hidden customer groups
- Useful for targeted marketing
- Supports business decision making

---

## Future Improvements

- Use PCA for better visualization
- Use Hierarchical Clustering
- Use DBSCAN for irregular clusters
- Build customer recommendation system
- Deploy dashboard using Streamlit
- Real-time customer segmentation

---
