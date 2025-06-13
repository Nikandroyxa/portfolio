# **Market Analysis - Customer Segmentation - RFM**

Coursework Mark for the report: 75%

Coursework Mark for the code: 79%

### Project Overview

This project focuses on analyzing customer purchasing behavior through **Market Basket Analysis** using RFM segmentation and clustering techniques. The analysis was performed on a dataset containing **38,765 purchase orders** from grocery stores over a two-year period. The primary goal was to understand customer behavior and identify key customer segments to enhance targeted marketing strategies and improve customer retention.

### **Process**

1. **Data Understanding and Cleaning**: Analyzed data distributions and performed transformations to handle skewness. Ensured data consistency and prepared the data for further analysis by managing redundant variables.
2. **RFM Segmentation**: Implemented an RFM (Recency, Frequency, Monetary) model using SQL to classify customers based on their recent transactions, frequency of purchases, and the total items bought. This segmentation enabled the identification of high-value customers and those requiring re-engagement strategies.
3. **Customer Clustering with DBSCAN**: Applied the DBSCAN algorithm for clustering customers into different tiers, based on their RFM scores. Used **knee-point method** to determine optimal parameters for clustering, ensuring accurate segment separation and cohesiveness.
4. **Cluster Profiling and Insights**: Interpreted clusters to identify valuable customer segments such as **High-Value Active Customers**, **Frequent Big Spenders**, and **Dormant Low-Spenders**. This analysis provided actionable insights for developing tailored marketing strategies.

### **Results**

The analysis revealed distinct customer segments:

- **High-Value VIPs**: Customers with high spending and frequent engagement.
- **Regular Spenders**: Consistent customers with a steady purchasing pattern.
- **Occasional Low-Spenders**: Customers with low engagement and spending.
- **Dormant Customers**: Customers with minimal interaction requiring reactivation efforts.

By identifying these segments, the marketing team can design personalized retention strategies to maximize **customer lifetime value** and **customer loyalty**.

### **Technologies Used**

- **SQL**: Data preparation, RFM calculation.
- **Python**: Clustering using DBSCAN, data transformation, and visualization.
- **Libraries**: Scikit-learn, Matplotlib, Pandas

