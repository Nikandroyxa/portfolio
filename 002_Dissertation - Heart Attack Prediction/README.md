# **Dissertation - Heart Attack Prediction**

Dissertation Mark for the report: 80%

Dissertation Mark for the code: 80%

### Project Overview

This project aimed to develop a Machine Learning model to predict heart attacks according Precision, Accuracy and AUC score and determine the top 10 features which influencing the target column.
We used two datasets for the experiments the "Heart Attack Risk Prediction" dataset, which includes 
8,763 synthetic patient records and 26 variables, and the "Heart Disease Health Indicators" dataset, which 
at first contained 253,661 synthetic instances with 22 features, and later was reduced to 50,000 instances 
for the flexibility.

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

