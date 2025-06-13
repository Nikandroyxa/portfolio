# **COVID-19 Monitoring Data Mart (ECDC)**

### Project Overview

This project focused on designing and implementing a strategic Data Mart for the European Centre for Disease Prevention and Control (ECDC) to support COVID-19 monitoring across EU/EEA countries. The aim was to build a dimensional data warehouse that integrates data from multiple healthcare processes and supports complex analytical queries for improved decision-making.

### **Process**

1. **Problem Specification**: Designed a solution based on five healthcare processes modeled with UML:
    - Quarantine
    - Vaccination
    - Infection
    - Drug Treatment
    - Citizen Recording  

2. **Dimensional Modeling**: Created an integrated Dimensional Fact Model (DFM) capable of answering key queries, such as:
    - Monthly COVID-19 cases, vaccinations, drug treatments, and quarantine numbers by venue and vaccine type
    - Country-wide monthly summaries

3. **Functional Dependencies & Attribute Tree**: Identified and listed functional dependencies from the relational schema. Constructed and pruned an attribute tree to identify relevant keys, facts, and dimensions for schema design.

4. **Fact Schema & Logical Model**: Mapped the conceptual DFM to a logical star schema:
    - One main fact table: `Fact_Covid_Analytics`
    - Dimensions: `Dim_Venue`, `Dim_Time`, `Dim_Vaccine`, `Dim_Treatment`, `Dim_Citizen`
    - Measures included new cases, ICU cases, deaths, vaccination count, quarantine count, and treatment count

5. **Implementation**: Developed and implemented the warehouse schema using SQL (MySQL or SQLite):
    - Wrote DDL statements to create dimension and fact tables
    - Ensured referential integrity and correct data types

6. **Materialized Views**: Created materialized views to improve performance of frequently executed analytical queries:
    - Monthly COVID-19 stats by venue and vaccine type
    - National-level monthly summaries for executive reporting

7. **OLAP Queries**: Wrote OLAP-style SQL queries to extract insights from the data:
    - Enabled slicing by venue, time, and vaccine
    - Supported roll-up and drill-down capabilities

### **Technologies Used**

- **SQL**: MySQL / SQLite
- **Dimensional M**
