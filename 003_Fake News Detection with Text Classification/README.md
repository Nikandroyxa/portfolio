# **Fake News Detection with Text Classification**

### **Project Overview**

This project aims to build a binary text classification model to distinguish between real and fake news articles. The dataset contains over 44,000 news articles across different domains (e.g., Politics, World News, US News). The primary goal was to evaluate classification performance across domains and identify key linguistic features driving model decisions.

The project was executed in two main phases: establishing a strong baseline model and then implementing targeted improvements to enhance performance.

---

### **Process**

1. **Data Preparation**  
    - Combined `Fake.csv` and `True.csv` from Kaggle  
    - Labeled fake as `0` and real as `1`  
    - Concatenated `title` and `text` into a unified `full` field  
    - Cleaned text (lowercasing, punctuation/digit removal, URL stripping)

2. **Baseline Modeling**  
    - Split dataset into stratified train/test sets  
    - Applied **TF-IDF** vectorization (unigrams, `max_df=0.8`)  
    - Trained **Logistic Regression** model  
    - Evaluated metrics:  
        - **Accuracy**: 98.76%  
        - **Precision**: 98.40%  
        - **Recall**: 99.00%  
        - **F1 Score**: 99.00%  
        - **AUC**: 0.9878  
    - Domain-level performance assessed using the `subject` feature  
    - Feature importance analysis revealed real vs fake linguistic signals

3. **Model Improvements**  
    - **TF-IDF with Bigrams**: Slight improvement, more contextual understanding  
    - **XGBoost Classifier**: Non-linear model yielded  
        - **Accuracy**: 99.78%  
        - **AUC**: 0.9978  
    - **Final Model with Categorical Feature (`subject`)**:  
        - One-hot encoded `subject` and combined with TF-IDF features  
        - Logistic Regression achieved **100% accuracy**, **F1**, and **AUC**  
        - Demonstrated the power of combining text with domain metadata

---

### **Technologies Used**

- **Python**: pandas, numpy, matplotlib, seaborn  
- **NLP**: Regex, string manipulation, TF-IDF Vectorization  
- **Modeling**: Logistic Regression, XGBoost  
- **Evaluation**: Accuracy, Precision, Recall, F1 Score, AUC, Confusion Matrix  
- **Feature Engineering**: Text cleaning, bigrams, domain metadata integration  


