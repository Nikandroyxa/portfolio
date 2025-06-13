#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import random
import mysql.connector
conn= mysql.connector.connect(user='root', password='Xa31121991!', host='localhost')
cur= conn.cursor()


# In[2]:


cur.execute('CREATE DATABASE IF NOT EXISTS CourseWork_Covid')
cur.execute('USE CourseWork_Covid')


# In[3]:


cur.execute('''CREATE TABLE DIM_CITIZEN
                ( CITIZEN_ID      INT(5) NOT NULL,
                  NAME            VARCHAR(50) NULL,
                  SURNAME         VARCHAR(50) NULL,
                  GENDER          VARCHAR(01) NULL,
                  DATE_OF_BIRTH   DATE NULL,
                  WEIGHT          FLOAT NULL,
                  HEIGHT          FLOAT NULL,
                  PHONE_NUMBER    VARCHAR(50) NULL,
                  ADDRESS         VARCHAR(50) NULL,
                  DISTRICT_ID     INT(5) NOT NULL,
                  PRIMARY KEY (CITIZEN_ID));
                ''')

sample_citizen_data= [(1, 'John', 'Doe', 'M', '1990-05-15', 75.5, 180.0, '123456789', '123 Main St', 1),
          (2, 'Jane', 'Smith', 'F', '1988-08-25', 65.2, 165.0, '987654321', '456 Elm St', 2),
          (3, 'Michael', 'Johnson', 'M', '1975-12-10', 80.0, 175.0, '5551234567', '789 Oak St', 3),
          (4, 'Emily', 'Davis', 'F', '1995-03-25', 55.0, 160.0, '1112223333', '321 Maple Ave', 1),
          (5, 'James', 'Wilson', 'M', '1983-09-12', 85.5, 185.0, '4445556666', '654 Pine St', 2),
          (6, 'Samantha', 'Brown', 'F', '1992-11-08', 60.0, 170.0, '9998887777', '987 Cedar St', 3),
          (7, 'Robert', 'Martinez', 'M', '1978-06-30', 70.0, 175.0, '3334445555', '555 Oak St', 1),
          (8, 'Amanda', 'Harris', 'F', '1993-04-17', 62.5, 160.0, '7778889999', '222 Elm St', 2),
          (9, 'Daniel', 'Garcia', 'M', '1987-10-05', 78.2, 180.0, '6667778888', '789 Pine St', 3),
          (10, 'Jessica', 'Taylor', 'F', '1991-02-20', 58.0, 165.0, '5556667777', '456 Maple Ave', 1),
          (11, 'William', 'Jones', 'M', '1982-07-15', 77.5, 175.0, '1112223333', '789 Elm St', 2),
          (12, 'Sarah', 'Johnson', 'F', '1994-09-10', 63.0, 170.0, '8889990000', '123 Pine St', 3),
          (13, 'Michael', 'Brown', 'M', '1980-03-25', 92.5, 190.0, '4445556666', '321 Oak St', 1),
          (14, 'Elizabeth', 'Wilson', 'F', '1990-11-08', 56.0, 160.0, '3334445555', '456 Cedar St', 2),
          (15, 'Kevin', 'Martinez', 'M', '1985-06-30', 68.0, 175.0, '9998887777', '789 Maple Ave', 3),
          (16, 'Melissa', 'Gonzalez', 'F', '1996-04-17', 64.2, 165.0, '7778889999', '222 Elm St', 1),
          (17, 'Daniel', 'Rodriguez', 'M', '1989-10-05', 81.3, 180.0, '5556667777', '789 Pine St', 2),
          (18, 'Jessica', 'Lopez', 'F', '1992-02-20', 59.0, 165.0, '6667778888', '456 Maple Ave', 3),
          (19, 'William', 'Hernandez', 'M', '1977-07-15', 79.5, 185.0, '1239876543', '987 Oak St', 1),
          (20, 'Jennifer', 'Flores', 'F', '1984-11-05', 73.0, 170.0, '9871234567', '321 Pine St', 2)

]

cur.executemany('''Insert Into DIM_CITIZEN (CITIZEN_ID, NAME, SURNAME, GENDER, DATE_OF_BIRTH, WEIGHT, 
                                            HEIGHT, PHONE_NUMBER, ADDRESS, DISTRICT_ID) 
                          VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''', sample_citizen_data)

q1= '''select *
       from DIM_CITIZEN'''
df1= pd.read_sql_query(q1, conn)
df1


# In[4]:


cur.execute('''CREATE TABLE DIM_TIME
                ( DATE_ID         INT(5) NOT NULL,
                  DATE            DATE NULL,
                  YEAR_ID         INT(5) NOT NULL,
                  MONTH_ID        INT(5) NOT NULL,
                  WEEK_ID         INT(5) NOT NULL,
                  DAY_OF_WEEK_ID  INT(5) NOT NULL,
                  PRIMARY KEY (DATE_ID));
                ''')

year = 2022

sample_time_data = []
for i in range(1, 21):
    month_id = random.randint(1, 12)
    date = f'{year}-{month_id:02d}-{i:02d}'
    week_id = (month_id - 1) * 4 + ((i - 1) // 7) + 1
    day_of_week_id = ((i - 1) % 7) + 1
    sample_time_data.append((i, date, year, month_id, week_id, day_of_week_id))

cur.executemany('''INSERT INTO DIM_TIME (DATE_ID, DATE, YEAR_ID, MONTH_ID, WEEK_ID, DAY_OF_WEEK_ID) 
                    VALUES (%s, %s, %s, %s, %s, %s)''', sample_time_data)

q1= '''select *
       from DIM_TIME'''
df1= pd.read_sql_query(q1, conn)
df1


# In[5]:


cur.execute('''CREATE TABLE DIM_QUARANTINE
                ( QUARANTINE_ID     INT(5) NOT NULL,
                  QUARANTINE_NAME   VARCHAR(50) NULL,
                  NUMBER_OF_DAYS    INT(5) NULL,
                  PRIMARY KEY (QUARANTINE_ID));
                ''')
sample_quarantine_data= [
    (1, 'Home Quarantine', 14),
    (2, 'Hotel Quarantine', 10),
    (3, 'Hospital Quarantine', 21),
    (4, 'Self-Isolation', 7)
    
]

cur.executemany('''INSERT INTO DIM_QUARANTINE (QUARANTINE_ID, QUARANTINE_NAME, NUMBER_OF_DAYS) 
                    VALUES (%s, %s, %s)''', sample_quarantine_data)

q1= '''select *
       from DIM_QUARANTINE'''
df1= pd.read_sql_query(q1, conn)
df1


# In[6]:


cur.execute('''CREATE TABLE DIM_VACCINE
                ( VACCINE_ID     INT(5) NOT NULL,
                  VACCINE_NAME   VARCHAR(50) NULL,
                  PRIMARY KEY (VACCINE_ID));
                ''')

sample_vaccine_data = [
    (1, 'Pfizer-BioNTech'),
    (2, 'Moderna'),
    (3, 'Johnson & Johnson'),
    (4, 'AstraZeneca')
]


cur.executemany('''INSERT INTO DIM_VACCINE (VACCINE_ID, VACCINE_NAME) 
                    VALUES (%s, %s)''', sample_vaccine_data)


# In[7]:


cur.execute('''CREATE TABLE DIM_HEALTH_UNIT
                ( HEALTH_UNIT_ID     INT(5) NOT NULL,
                  HEALTH_UNIT_NAME   VARCHAR(50) NULL,
                  ADDRESS            VARCHAR(50) NULL,
                  DISTRICT_ID        INT(5) NOT NULL,
                  PRIMARY KEY (HEALTH_UNIT_ID));
                ''')

sample_health_unit_data = [
    (1, 'City Hospital', '123 Main Street', 1),
    (2, 'Community Clinic', '456 Elm Street', 2),
    (3, 'Regional Medical Center', '789 Oak Street', 3)
]


cur.executemany('''INSERT INTO DIM_HEALTH_UNIT (HEALTH_UNIT_ID, HEALTH_UNIT_NAME, ADDRESS, DISTRICT_ID) 
                    VALUES (%s, %s, %s, %s)''', sample_health_unit_data)


# In[8]:


cur.execute('''CREATE TABLE DIM_DISEASE_VARIANT
                ( DISEASE_VARIANT_ID     INT(5) NOT NULL,
                  DISEASE_VARIANT_NAME   VARCHAR(50) NULL,
                  DISEASE_ID             INT(5) NOT NULL,
                  PRIMARY KEY (DISEASE_VARIANT_ID));
                ''')

sample_disease_variant_data = [
    (1, 'Variant A', 1),
    (2, 'Variant B', 2),
    (3, 'Variant C', 3)
]


cur.executemany('''INSERT INTO DIM_DISEASE_VARIANT (DISEASE_VARIANT_ID, DISEASE_VARIANT_NAME, DISEASE_ID) 
                    VALUES (%s, %s, %s)''', sample_disease_variant_data)


# In[9]:


cur.execute('''CREATE TABLE DIM_PATIENT_STATUS
                ( PATIENT_STATUS_ID     INT(5) NOT NULL,
                  PATIENT_STATUS_NAME   VARCHAR(50) NULL,
                  PRIMARY KEY (PATIENT_STATUS_ID));
                ''')

sample_patient_status_data = [
    (1, 'Stable'),
    (2, 'Critical'),
    (3, 'Serious')
]


cur.executemany('''INSERT INTO DIM_PATIENT_STATUS (PATIENT_STATUS_ID, PATIENT_STATUS_NAME) 
                    VALUES (%s, %s)''', sample_patient_status_data)


# In[10]:


cur.execute('''CREATE TABLE DIM_DRUG
                ( DRUG_ID      INT(5) NOT NULL,
                  DRUG_NAME    VARCHAR(50) NULL,
                  FOR_DEISEASE VARCHAR(50) NULL,
                  PRIMARY KEY (DRUG_ID));
                ''')

sample_drug_data = [
    (1, 'Paracetamol', 'Fever'),
    (2, 'Ibuprofen', 'Pain'),
    (3, 'Biofenax', 'Μotion Sickness')
]


cur.executemany('''INSERT INTO DIM_DRUG (DRUG_ID, DRUG_NAME, FOR_DEISEASE) 
                    VALUES (%s, %s, %s)''', sample_drug_data)


# In[11]:


cur.execute('''CREATE TABLE DIM_VENUE
                ( VENUE_ID      INT(5) NOT NULL,
                  VENUE_TYPE    VARCHAR(50) NULL,
                  VENUE_ADDRESS VARCHAR(50) NULL,
                  PRIMARY KEY (VENUE_ID));
                ''')

sample_venue_data = [
    (1, 'Hospital', '123 Main Street'),
    (2, 'Community Center', '456 Elm Street'),
    (3, 'School', '789 Oak Street')
]

cur.executemany('''INSERT INTO DIM_VENUE (VENUE_ID, VENUE_TYPE, VENUE_ADDRESS) 
                    VALUES (%s, %s, %s)''', sample_venue_data)


# In[12]:


cur.execute('''CREATE TABLE FACT_CITIZEN_QUARANTINE
                ( CITIZEN_ID      INT(5) NOT NULL,
                  QUARANTINE_ID   INT(5) NOT NULL,
                  START_DATE_ID   INT(5) NOT NULL,
                  END_DATE_ID     INT(5) NOT NULL,
                  No_Of_QUARANTINE INT(5) NOT NULL,
                  PRIMARY KEY (CITIZEN_ID, QUARANTINE_ID, START_DATE_ID, END_DATE_ID),
                  FOREIGN KEY (CITIZEN_ID) REFERENCES DIM_CITIZEN (CITIZEN_ID),
                  FOREIGN KEY (QUARANTINE_ID) REFERENCES DIM_QUARANTINE (QUARANTINE_ID),
                  FOREIGN KEY (START_DATE_ID) REFERENCES DIM_TIME (DATE_ID),
                  FOREIGN KEY (END_DATE_ID) REFERENCES DIM_TIME (DATE_ID)
                  );
                ''')


# In[13]:


cur.execute('''CREATE TABLE FACT_CITIZEN_VACCINES
                ( CITIZEN_ID      INT(5) NOT NULL,
                  VACCINE_ID      INT(5) NOT NULL,
                  DATE_ID         INT(5) NOT NULL,
                  HEALTH_UNIT_ID  INT(5) NOT NULL,
                  PRIMARY KEY (CITIZEN_ID, VACCINE_ID, DATE_ID, HEALTH_UNIT_ID),
                  FOREIGN KEY (CITIZEN_ID) REFERENCES DIM_CITIZEN (CITIZEN_ID),
                  FOREIGN KEY (VACCINE_ID) REFERENCES DIM_VACCINE (VACCINE_ID),
                  FOREIGN KEY (DATE_ID) REFERENCES DIM_TIME (DATE_ID),
                  FOREIGN KEY (HEALTH_UNIT_ID) REFERENCES DIM_HEALTH_UNIT (HEALTH_UNIT_ID)
                  );
                ''')


# In[14]:


cur.execute('''CREATE TABLE FACT_PATIENT
                ( PATIENT_ID          INT(5) NOT NULL,
                  DISEASE_VARIANT_ID  INT(5) NOT NULL,
                  DATE_ID             INT(5) NOT NULL,
                  PATIENT_STATUS_ID   INT(5) NOT NULL,
                  PRIMARY KEY (PATIENT_ID, DISEASE_VARIANT_ID, DATE_ID, PATIENT_STATUS_ID),
                  FOREIGN KEY (PATIENT_ID) REFERENCES DIM_CITIZEN (CITIZEN_ID),
                  FOREIGN KEY (DISEASE_VARIANT_ID) REFERENCES DIM_DISEASE_VARIANT (DISEASE_VARIANT_ID),
                  FOREIGN KEY (DATE_ID) REFERENCES DIM_TIME (DATE_ID),
                  FOREIGN KEY (PATIENT_STATUS_ID) REFERENCES DIM_PATIENT_STATUS (PATIENT_STATUS_ID)
                  );
                ''')


# In[15]:


cur.execute('''CREATE TABLE FACT_PATIENT_DRUG
                ( PATIENT_ID      INT(5) NOT NULL,
                  DRUG_ID         INT(5) NOT NULL,
                  START_DATE_ID   INT(5) NOT NULL,
                  END_DATE_ID     INT(5) NOT NULL,
                  DOSE            VARCHAR(50) NULL,
                  PRIMARY KEY (PATIENT_ID, DRUG_ID, START_DATE_ID, END_DATE_ID),
                  FOREIGN KEY (PATIENT_ID) REFERENCES DIM_CITIZEN (CITIZEN_ID),
                  FOREIGN KEY (DRUG_ID) REFERENCES DIM_DRUG (DRUG_ID),
                  FOREIGN KEY (START_DATE_ID) REFERENCES DIM_TIME (DATE_ID),
                  FOREIGN KEY (END_DATE_ID) REFERENCES DIM_TIME (DATE_ID)
                  );
                ''')


# In[16]:


cur.execute('''CREATE TABLE FACT_CITIZEN_VENUES
                ( CITIZEN_ID      INT(5) NOT NULL,
                  VENUE_ID        INT(5) NOT NULL,
                  DATE_ID         INT(5) NOT NULL,
                  PRIMARY KEY (CITIZEN_ID, VENUE_ID, DATE_ID),
                  FOREIGN KEY (CITIZEN_ID) REFERENCES DIM_CITIZEN (CITIZEN_ID),
                  FOREIGN KEY (VENUE_ID) REFERENCES DIM_VENUE (VENUE_ID),
                  FOREIGN KEY (DATE_ID) REFERENCES DIM_TIME (DATE_ID)
                  );
                ''')


# In[17]:


conn.commit()
conn.close()


# In[18]:


conn= mysql.connector.connect(user='root', password='Xa31121991!', host='localhost', database='CourseWork_Covid')
cur= conn.cursor()


# In[21]:



q1= '''SELECT SUM(No_Of_QUARANTINE), dt.DATE_ID
FROM FACT_CITIZEN_QUARANTINE fcq
JOIN DIM_CITIZEN dc ON fcq.CITIZEN_ID = dc.CITIZEN_ID
JOIN DIM_TIME dt ON fcq.START_DATE_ID = dt.DATE_ID AND fcq.END_DATE_ID = dt.DATE_ID
JOIN DIM_QUARANTINE dq ON fcq.QUARANTINE_ID = dq.QUARANTINE_ID
GROUP BY dt.DATE_ID;'''

df1= pd.read_sql_query(q1, conn)
df1


# In[22]:


q1 = '''SELECT * FROM FACT_CITIZEN_QUARANTINE'''
df1 = pd.read_sql_query(q1, conn)
print(df1.head())


# In[23]:


q2 = '''SELECT * FROM DIM_TIME'''
df2 = pd.read_sql_query(q2, conn)
print(df2.head())


# In[24]:


q3 = '''SELECT * FROM DIM_CITIZEN'''
df3 = pd.read_sql_query(q3, conn)
print(df3.head())


# In[25]:


q4 = '''SELECT * FROM DIM_QUARANTINE'''
df4 = pd.read_sql_query(q4, conn)
print(df4.head())


# In[27]:


print(cur.execute('''INSERT INTO FACT_CITIZEN_QUARANTINE (CITIZEN_ID, QUARANTINE_ID, START_DATE_ID, END_DATE_ID, No_Of_QUARANTINE) 
                    VALUES (%s, %s, %s, %s, %s)''', sample_fact_data))


# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


create_view_query1 = """
CREATE OR REPLACE VIEW View_Query1 AS
SELECT
    fv.VENUE_ID,
    dt.MONTH_ID,
    COUNT(*) AS COVID19_Cases,
    SUM(CASE WHEN fcv.VACCINE_ID IS NOT NULL THEN 1 ELSE 0 END) AS Vaccination_Count,
    COUNT(DISTINCT fq.QUARANTINE_ID) AS Quarantine_Count,
    COUNT(DISTINCT fd.DRUG_ID) AS Drug_Treatment_Count
FROM
    FACT_CITIZEN_VENUES fv
JOIN
    DIM_TIME dt ON fv.DATE_ID = dt.DATE_ID
LEFT JOIN
    FACT_CITIZEN_QUARANTINE fq ON dt.DATE_ID BETWEEN fq.START_DATE_ID AND fq.END_DATE_ID
LEFT JOIN
    FACT_PATIENT_DRUG fd ON dt.DATE_ID BETWEEN fd.START_DATE_ID AND fd.END_DATE_ID
LEFT JOIN
    FACT_CITIZEN_VACCINES fcv ON fv.CITIZEN_ID = fcv.CITIZEN_ID AND fv.DATE_ID = fcv.DATE_ID
GROUP BY
    fv.VENUE_ID, dt.MONTH_ID;
"""

cur.execute(create_view_query1)

print("View created successfully!")


# In[20]:


create_view_query2 = """
CREATE OR REPLACE VIEW View_Query2 AS
SELECT
    dt.MONTH_ID,
    COUNT(*) AS COVID19_Cases,
    COUNT(DISTINCT fq.QUARANTINE_ID) AS Quarantine_Count,
    COUNT(DISTINCT fd.DRUG_ID) AS Drug_Treatment_Count
FROM
    DIM_TIME dt
LEFT JOIN
    FACT_PATIENT fp ON dt.DATE_ID = fp.DATE_ID
LEFT JOIN
    FACT_PATIENT_DRUG fd ON fp.PATIENT_ID = fd.PATIENT_ID AND dt.DATE_ID BETWEEN fd.START_DATE_ID AND fd.END_DATE_ID
LEFT JOIN
    FACT_CITIZEN_QUARANTINE fq ON dt.DATE_ID BETWEEN fq.START_DATE_ID AND fq.END_DATE_ID
GROUP BY
    dt.MONTH_ID;
"""
cur.execute(create_view_query2)


conn.commit()
conn.close()

print("View created successfully!")


# In[21]:


conn= mysql.connector.connect(user='root', password='Xa31121991!', host='localhost', database='CourseWork_Covid')
cur= conn.cursor()


# In[22]:


select_query = "SELECT * FROM View_Query2;"

cur.execute(select_query)

results = cur.fetchall()

for row in results:
    print(row)

cur.close()
conn.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




