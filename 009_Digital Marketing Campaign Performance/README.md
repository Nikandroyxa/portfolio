# **Digital Marketing Performance Analysis**

### **Project Overview**

This project focused on analyzing and visualizing the performance of marketing campaigns using SQL and Tableau. The goal was to calculate key marketing KPIs and generate actionable insights to optimize advertising strategy.
We used a dataset from [Kaggle](https://www.kaggle.com/datasets/sinderpreet/analyze-the-marketing-spending/data) containing 308 marketing campaign records, including details like impressions, clicks, leads, orders, spend, and revenue across various digital channels. This self-driven project allowed me to demonstrate my SQL, analytical, and data visualization skills by simulating a real-world marketing analytics scenario—directly


### **Process**

1. **Data Preparation**  
   - Imported and cleaned the raw `CSV` data using **MySQL Workbench**
   - Created a relational table schema with appropriate data types for marketing campaign fields
   - Verified successful data load using exploratory SQL queries

2. **Metric Calculations (KPIs)**  
   Developed SQL queries to compute core digital marketing performance indicators:
   - **ROMI**: Return on Marketing Investment  
   - **CTR**: Click-Through Rate  
   - **Conversion Rates** (click → lead, lead → order)  
   - **AOV**: Average Order Value  
   - **CPC / CPL / CAC**: Cost per Click, Cost per Lead & Customer Acquisition Cost Acquisition  

3. **Aggregated Insights**  
   - Calculated average ROMI by Campaign & Category
   - Identified daily trends in Revenue vs Marketing Spend
   - Aggregated Average CPC, CPL & CAC by Campaign category
   - Evaluated campaign activity across weekdays

4. **Dashboard Development in Tableau**  
   Exported aggregated results and visualized them in an interactive Tableau dashboard:
   - Weekly trends of marketing Spend vs Revenue
   - Grouped bar chart comparing CPC / CPL / CAC by Category
   - Top 10 Campaigns by Average ROMI
   - ROMI by Campaign Category


### **Key Metrics & Visuals**

📊 **Charts Included in the Dashboard**:

1. **Revenue vs. Spend (Weekly)**  
   Visualizes how marketing investments aligned with revenue performance over time.

2. **CPC / CPL / CAC by Category**  
   Helps compare the cost-efficiency of influencer, search, media, and social campaigns.

3. **Top Campaigns by ROMI**  
   Highlights campaigns that returned the most value per dollar spent.

4. **ROMI by Category**  
   Compares performance across major marketing channels.

🧭 **Interactive Dashboard**:  
👉 [View on Tableau Public](https://public.tableau.com/views/MarketingCampaignPerformanceDashboard_17511978417850/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)


### **Technologies Used**

- **SQL (MySQL Workbench)** – data modeling, cleaning, KPI calculations  
- **Tableau Public** – dashboard creation & visualization  
- **Excel** – interim export of query results  
- **Kaggle Dataset** – digital marketing campaign source
