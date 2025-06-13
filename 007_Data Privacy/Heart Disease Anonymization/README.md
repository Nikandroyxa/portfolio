# **Heart Disease Dataset Anonymization Project**

### **Project Overview**

This project explores industry-standard data anonymization techniques applied to the UCI Heart Disease dataset. Inspired by research into the responsibilities of a Data Privacy Scientist, the aim was to gain hands-on experience with techniques like **k-anonymity**, **l-diversity**, and **t-closeness**, and apply them to a real dataset to minimize re-identification risk while maintaining analytical utility.

### **Process**

1. **Data Understanding & Preprocessing**  
    - Selected relevant QIDs: `age`, `sex`, `cp`  
    - Standardized string formatting for categorical variables  
    - Generalized:
        - `age` → `age_group`  
        - `cp` → `cp_generalized`  
        - `num` → `num_gen` (binary: 0 for no disease, 1 for any heart disease)

2. **k-Anonymity**  
    - Combined QIDs into a single `QID` column  
    - Evaluated minimum group size = 7  
    - Ensured every individual shared their profile with at least 6 others (`k ≥ 2`)

3. **l-Diversity**  
    - Checked how many distinct `num_gen` values each QID group had  
    - Found that 98.6% of records met `l ≥ 2`  
    - Removed 13 records violating l-diversity to prevent attribute disclosure

4. **t-Closeness**  
    - Compared sensitive attribute distribution per group to overall dataset  
    - Used sum of absolute differences as a proxy for Earth Mover’s Distance  
    - Flagged groups with `t-closeness > 0.4` as high-risk  
    - Removed those high-risk QID groups to minimize statistical inference risks

### **Technologies Used**

- **Python**: Pandas, NumPy, Matplotlib, Seaborn  
- **Privacy Techniques**: k-Anonymity, l-Diversity, t-Closeness  
- **Data Transformation**: Generalization of categorical and numeric fields  
- **Risk Assessment**: Group-based frequency distribution analysis  


