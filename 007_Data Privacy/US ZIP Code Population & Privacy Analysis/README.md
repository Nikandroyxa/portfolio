# **US ZIP Code Population & Privacy Analysis**

### **Project Overview**

This project analyzes US ZIP code-level population and land data to identify privacy risks, inconsistencies, and population density extremes. Inspired by real-world tasks assigned to Privacy Data Scientists, it simulates a practical exercise. The focus is on identifying ZIP-level uniqueness, aggregating ZIP5 to ZIP3 codes, and exploring distributions relevant to privacy protection and data generalization.

### **Process**

1. **Data Understanding & Preprocessing**  
    - Loaded ZIP5-level data (`df_a`) and ZIP code–to–state mapping (`df_z`)  
    - Standardized ZIP codes to 5-digit string format  
    - Converted land area from m² to km² and cleaned columns with extensive nulls  
    - Resolved minor missing values (e.g., for ZIP codes `96860` and `96863` in Hawaii)

2. **ZIP Code Analysis**  
    - Found the state with the most ZIP codes  
    - Identified the most easterly, westerly (excluding Alaska), and northerly ZIP codes  
    - Calculated population density and visualized the top 5 most densely populated ZIP codes

3. **ZIP3 Consistency & Privacy Check**  
    - Created a 3-digit ZIP code (`zip3c`) column  
    - Identified ZIP3 codes common to more than one state (`063`, `834`)  
    - Flagged and listed incongruous places (cities existing in overlapping states)

4. **3-Digit ZIP Code Aggregation**  
    - Excluded ambiguous ZIP3s and aggregated population and land area by ZIP3 and state  
    - Counted how many ZIP3s had population under 20,000  
    - Plotted density distributions using:
        - Boxplot with log scale  
        - Raw and log-scaled histograms with KDE

5. **Sparse Density Exploration**  
    - Located the ZIP3 with the lowest population density (in **Wyoming**)  
    - Discussed findings in context of U.S. geography and demographics

6. **Population Distribution Comparison**  
    - Compared population density distributions between ZIP5 and ZIP3 codes  
    - Highlighted trade-offs in granularity and generalization  
    - Used log-scaled histograms for clearer comparison

### **Technologies Used**

- **Python**: Pandas, NumPy, Seaborn, Matplotlib  
- **Data Cleaning**: Null handling, standardization, aggregation  
- **Privacy Concepts**: ZIP-level overlap detection, low-density detection, ZIP-level generalization  
- **Visualization**: Boxplots, histograms, KDE plots, log-scaling for skewed data

