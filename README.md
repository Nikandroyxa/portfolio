# Projects

[Babis Nikandrou CV DS_Mar 25.pdf](attachment:aa8d6e21-01e7-440f-8e23-518a1b1f27b9:Babis_Nikandrou_CV_DS_Mar_25.pdf)

## **Babis Nikandrou**

Welcome to my portfolio! 

I have worked on various Data Science Projects, where I have used **Python**, **SQL**, **Data Preparation**, **Data Cleansing**, **Data Warehousing**, **ETL** and **Machine Learning** libraries like **Pandas**, **Numpy**, **Seaborn**, **Matplotlib**, **Scikit-learn**, **XGBoost**, and **NLTK**. 

I have worked in real life scenarios such as building **Predictive Models** for **Heart Attack Prediction** and **Monkeypox Screening** and conducting  **RFM Segmentation** and **Clustering** in customer segmentation where I've applied advanced techniques like **Grid Search Cross-Validation**, **DBSCAN Clustering**, and **Latent Dirichlet Allocation (LDA)**. I’ve also worked with **APIs** and **Text Mining** tools to analyze social media data, and SQL for data management and analysis in large-scale competitions such as LEGO Education.

# **Dissertation - Heart Attack Prediction**

Dissertation Mark for the report: 80%

Dissertation Mark for the code: 80%

### Project Overview

This project aimed to develop a Machine Learning model to predict heart attacks according Precision, Accuracy and AUC score and determine the top 10 features which influencing the target column.

### Process

- **Data Cleaning**: Addressed missing values and ensured data quality for accurate analysis.
- **Data Scaling**: Applied Standard Scaling and Robust Scaling (for skewed data) to normalize features.
- **Machine Learning Algorithms**:
    - Logistic Regression (LR)
    - Decision Tree (DT)
    - K-Nearest Neighbors (KNN)
    - Naïve Bayes (NB)
    - Support Vector Machine (SVM)
    - Random Forest (RF)
    - Gradient Boosting (XGBoost, LGBM, CatBoost)
    - Adaptive Boosting (AdaBoost)
- **Imbalance Handling**:
    - **Over-Sampling**: SMOTE, ADASYN, Random Over Sampler (ROS)
    - **Under-Sampling**: Random Under Sampler (RUS), Tomek Links, Cluster Centroids
    - **Combination Methods**: SMOTETomek, CC & SMOTE
    - **One-Class Methods**: One-Class SVM, Isolation Forest
    - **Cost-Sensitive and Calibration Methods**: To refine model predictions.
- **Feature Selection**
    - **Filter Methods**: ANOVA, Chi-Square, Mutual Information
    - **Wrapper Methods**: Recursive Feature Elimination (RFE), Forward and Backward Selection
    - **Embedded Methods**: Lasso Regression
- **Model Tuning**

The four best-performing models were further optimized using **Grid Search Cross-Validation** and **Randomized Search Cross-Validation** to enhance their performance.

### **Technologies Used**

- **Python**: Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, CatBoost
- **Data Visualization**: Matplotlib, Seaborn
- **Imbalance Techniques**: Imbalanced-learn library

# **Monkeypox (MPOX)**

Coursework Mark for the report: 83%

Coursework Mark for the code: 89%

### Project Overview

This project focused on building a Machine learning model to predict Monkeypox (MPOX) infection, aiming to reduce the reliance on costly and time-consuming PCR tests. The goal was to develop an accurate, efficient, and inexpensive predictive model that could identify individuals likely to have contracted the virus without needing a PCR test.

### **Process**

1. **Data Understanding**: Utilized a dataset containing historical PCR test results, with features like age, symptoms (e.g., sore throat, oral lesions), and HIV infection status etc.
2. **Data Cleaning and Transformation**: Addressed missing values, corrected data types, and transformed variables for analysis. Key preprocessing steps included converting categorical variables into numerical formats and managing outliers.
3. **Modeling**:
    - Logistic Regression (LR)
    - Decision Tree (DT)
    - K-Nearest Neighbors (KNN)
    - Support Vector Machine (SVM) with RBF kernel
    - Naïve Bayes (NB) Models were optimized using GridSearchCV and pre-pruning techniques to enhance performance.
4. **Evaluation Metrics:**

Precision, Recall, and F1-Score:

### **Technologies Used**

- **Python**: Pandas, Scikit-learn, Matplotlib
- **Modeling Techniques**: Logistic Regression, Decision Trees, KNN, SVM, Naïve Bayes
- **Model Optimization**: GridSearchCV, Pre-pruning

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

# Social Media Analysis

Coursework Mark for the report: 95%

Coursework Mark for the code: 92%

### Project Overview

This project aimed to investigate the experiences of electric motorcycle owners using social media data. The focus was on analyzing discussions surrounding **Harley-Davidson’s LiveWire** model. By using **Text Mining techniques**, this project sought to understand the opinions, sentiments, and key themes expressed by electric vehicle owners, providing valuable insights into the industry.

### **Process**

1. **Data Collection**: Gathered social media data using **YouTube Data API** and **Reddit API** to collect comments and posts discussing Harley-Davidson’s LiveWire. Search terms included keywords like "Harley-Davidson LiveWire" and "electric motorcycle review."
2. **Data Preprocessing**: Cleaned the data by removing duplicate comments, special characters, and irrelevant information. Stored the data in structured formats (CSV) for further analysis.
3. **Exploratory Analysis**: Identified common words, phrases, and discussion patterns using Python libraries. Visualized the frequency of key terms such as "bike," "electric," and "LiveWire," providing an understanding of popular topics.
4. **Text Mining Techniques**: Applied **Latent Dirichlet Allocation (LDA)** for topic modeling and **VADER Sentiment Analysis** to uncover the sentiment behind discussions. Topics ranged from general experiences with electric motorcycles to pricing concerns and performance considerations.

### **Results**

- **Common Themes**: Discussions frequently focused on **performance**, **battery life**, and **pricing** of electric motorcycles.
- **Sentiment Analysis**: The overall sentiment was **slightly positive**, with enthusiasm for electric motorcycles’ benefits but some negativity concerning high prices.
- **Key Discussions**: Topic modeling identified themes like preferences for specific models, discussions on battery technology, and comparisons with traditional motorcycles.

### **Technologies Used**

- **Python**: Pandas, Scikit-learn, Matplotlib, NLTK (for sentiment analysis)
- **APIs**: YouTube Data API, Reddit API for data extraction
- **Text Mining**: LDA for topic modeling, VADER for sentiment analysis

# **LEGO Education of Greece- O3 Competition**

### **Project Overview**

This project focused on analyzing the data from the annual LEGO Education of Greece competition, hosted by O3. The event attracted 400 teams, comprising approximately 3,500 children aged 8 to 16, over a three-day period. The analysis aimed to gain insights into the demographics, activities, and engagement of the participants to enhance future events and improve overall experience.

### **Process**

1. **Data Collection**
2. **Data Management with SQL**
    - **Data Cleaning**: Identified and corrected inconsistencies, missing values, and outliers.
    - **Data Aggregation**: Aggregated data by age groups, gender, and location to identify trends and participation patterns.
    - **Query Optimization**: Optimized SQL queries to enhance the speed and performance of data retrieval for reporting and analysis.

### **Analysis**

- **Demographic Insights**: Distribution of participants by age, gender, and region, helping identify the most engaged age groups and areas with the highest interest in STEM education.
- **Activity Analysis**: Assessed which activities and challenges attracted the most participation, helping to refine the design of future competitions.
- **Engagement Trends**: Analyzed team performance and engagement levels across different age groups, providing insights into how to tailor activities to different skill levels.

### **Results**

- Optimize the structure and content of future competitions to better align with the interests and abilities of participants.
- Improve regional outreach strategies by targeting areas with lower participation.
- Design activities that cater to the diverse age groups involved, ensuring an inclusive and enriching experience for all participants.

### **Technologies Used**

- **SQL**: Data cleaning, aggregation, and analysis.
- **Excel**: Data visualization and reporting.
- **LEGO Event Data Platform**: Managed participant registration and activity tracking data.
