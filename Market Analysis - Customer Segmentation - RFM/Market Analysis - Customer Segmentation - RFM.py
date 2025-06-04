#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df= pd.read_csv("C://Users//User//Desktop//MSc Westminster//Data Warehousing and Business Intelligence//CourseWork_2//Basket_dataset.csv")
df.head(5)


# In[2]:


df.shape


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.info()


# In[ ]:





# ---  Statistcal & Distribution Analysis ---

# --- Item Description Analysis ---

# In[6]:


item_counts= df['itemDescription'].value_counts().sort_values(ascending=False)
item_counts.describe()


# In[7]:


plt.figure(figsize=(10, 6))
sns.set(style = 'whitegrid')
sns.distplot(item_counts)
plt.title('Distribution of Item Description', fontsize = 20)
plt.xlabel('Item Description')
plt.ylabel('Count')


# --- Members Analysis ---

# In[8]:


customer_item_counts = df.groupby('Member_number')['itemDescription'].count().sort_values(ascending=False)
customer_item_counts.describe()


# In[9]:


plt.figure(figsize=(10, 6))
sns.set(style = 'whitegrid')
sns.distplot(customer_item_counts)
plt.title('Distribution of Members vs Total Items', fontsize = 20)
plt.xlabel('No of Items')
plt.ylabel('Count')


# In[10]:


unique_customers_per_day = df.groupby('Date')['Member_number'].nunique().sort_values(ascending=False)
unique_customers_per_day.describe()


# In[11]:


plt.figure(figsize=(10, 6))
sns.set(style = 'whitegrid')
sns.distplot(unique_customers_per_day)
plt.title('Distribution of Members vs Date', fontsize = 20)
plt.xlabel('No of Members')
plt.ylabel('Count')


# In[12]:


no_of_items_per_member= df.groupby(df['Member_number']).size().reset_index(name= 'No_of_items')
no_of_items_per_member_per_busket= df.groupby([df['Member_number'], df['Date'].dt.date]).size().reset_index(name='No_of_items')
busket_per_member= df.groupby([df['Member_number'], df['Date'].dt.date])['Date'].nunique().reset_index(name='No_of_Buskets')
no_of_buskets_per_member= busket_per_member.groupby('Member_number')['Date'].nunique().reset_index(name='No_of_Buskets')
no_of_buskets_and_no_of_items_per_member= pd.merge(no_of_buskets_per_member, no_of_items_per_member, on='Member_number')
no_of_buskets_and_no_of_items_per_member['Avg_itmes']= (no_of_buskets_and_no_of_items_per_member['No_of_items']/no_of_buskets_and_no_of_items_per_member['No_of_Buskets']).round(0)
no_of_buskets_and_no_of_items_per_member.head(5)


# In[13]:


no_of_buskets_and_no_of_items_per_member['No_of_Buskets'].describe()


# In[14]:


plt.figure(figsize=(10, 6))
sns.set(style = 'whitegrid')
sns.distplot(no_of_buskets_and_no_of_items_per_member['No_of_Buskets'])
plt.title('Distribution of Buskets', fontsize = 20)
plt.xlabel('No of Buskets')
plt.ylabel('Count')


# --- Association identification  ---

# In[15]:


df1= df
df1['transaction']= df1['Member_number'].astype(str)+'_'+df1['Date'].astype(str)
df1.head()


# In[16]:


df1= pd.crosstab(df1['transaction'], df1['itemDescription'])
df1.head(5)


# In[17]:


def encode(item_freq):
    res = 0
    if item_freq > 0:
        res = 1
    return res
    
basket_transform= df1.applymap(encode)
basket_transform.head(5)


# In[18]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

freq_items= apriori(basket_transform, min_support=0.001, use_colnames=True)
associations= association_rules(freq_items, metric="lift")
associations.head(5)


# In[19]:


associations.sort_values(["support", "confidence", "lift"], axis=0, ascending=False).head(10)


# In[ ]:





# --- RFM ---

# In[20]:


import sqlite3
con= sqlite3.connect('Test')
cur= con.cursor()


# In[21]:


df.to_sql('Basket_dataset', con)


# In[22]:


cleandata= pd.read_sql(''' SELECT Member_number
                               ,MAX(Date) AS last_order_date
                               ,COUNT(DISTINCT Date) AS no_of_buskets
                               ,COUNT(*) AS no_of_items
                           FROM Basket_dataset
                           GROUP BY Member_number''', con)
cleandata.head(5)


# In[23]:


cleandata.to_sql("cleandata", con)


# In[24]:


cleandata.dtypes


# In[25]:


import numpy as np
import datetime
cleandata['last_order_date'] = pd.to_datetime(cleandata['last_order_date'])
cleandata.dtypes


# In[26]:


snapshot_date= cleandata['last_order_date'].max() + datetime.timedelta(days=1)
print(snapshot_date)


# In[27]:


customers= cleandata.groupby(['Member_number']).agg({
                                                       'last_order_date': lambda x: (snapshot_date - x.max()).days,
                                                       'no_of_buskets':'sum',
                                                       'no_of_items': 'sum'
                                                        })


# In[28]:


customers.rename(columns= {'last_order_date': 'Recency',
                            'no_of_buskets': 'Frequency',
                            'no_of_items': 'MonetaryValue'}, inplace=True)


# In[29]:


customers.head(5)


# In[30]:


customers.to_sql("customers", con)


# ---  DBSCAN  ---

# In[31]:


customers.head(5)


# In[32]:


plt.figure(figsize=(12,10))
plt.subplot(3, 1, 1); sns.distplot(customers['Recency'])
plt.subplot(3, 1, 2); sns.distplot(customers['Frequency'])
plt.subplot(3, 1, 3); sns.distplot(customers['MonetaryValue'])

plt.show()


# In[33]:


from scipy import stats
customers_fix= pd.DataFrame()
customers_fix["Recency"]= stats.boxcox(customers['Recency'])[0]
customers_fix["Frequency"]= stats.boxcox(customers['Frequency'])[0]
customers_fix["MonetaryValue"]= stats.boxcox(customers['MonetaryValue'])[0]
customers_fix.tail()


# In[34]:


plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1); sns.distplot(customers_fix['Recency'])
plt.subplot(3, 1, 2); sns.distplot(customers_fix['Frequency'])
plt.subplot(3, 1, 3); sns.distplot(customers_fix['MonetaryValue'])

plt.show()


# --- Normalisation ---

# In[35]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit(customers_fix)
customers_normalized= scaler.transform(customers_fix)

print(customers_normalized.mean(axis= 0).round(2))
print(customers_normalized.std(axis= 0).round(2))


# In[36]:


customers_normalized_df= pd.DataFrame(customers_normalized, columns= customers_fix.columns)
customers_normalized_df.head(5)


# In[37]:


from sklearn.neighbors import NearestNeighbors

neighbors= NearestNeighbors(n_neighbors= 6)
neighbors_fit= neighbors.fit(customers_normalized_df)
distances, indices= neighbors_fit.kneighbors(customers_normalized_df)


# In[38]:


distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)


# In[39]:


from kneed import KneeLocator
kneedle= KneeLocator(x = range(1, len(distances)+1), y= distances, S= 1.0,
                      curve= "concave", direction= "increasing", online= True)

kneedle.plot_knee()
plt.show()
print('epsilon=',round(kneedle.knee_y,2))


# In[40]:


from sklearn.cluster import DBSCAN

db= DBSCAN(eps= 0.35, min_samples= 6).fit(customers_normalized_df)

customers_normalized_df['Labels']= db.labels_
plt.figure(figsize=(11, 6))
sns.scatterplot(x= "Recency", y= "Frequency", data= customers_normalized_df, hue= customers_normalized_df.Labels,
                palette= sns.color_palette('hls', np.unique(db.labels_).shape[0]))

plt.show()


# In[41]:


from mpl_toolkits.mplot3d import Axes3D
db= DBSCAN(eps= 0.35, min_samples= 6).fit(customers_normalized_df[['Recency', 'Frequency', 'MonetaryValue']])
customers_normalized_df['Labels']= db.labels_
fig= plt.figure(figsize=(11, 6))
ax= fig.add_subplot(111, projection= '3d')
for label in np.unique(db.labels_):
    cluster= customers_normalized_df[customers_normalized_df['Labels']== label]
    ax.scatter(cluster['Recency'], cluster['Frequency'], cluster['MonetaryValue'], label= label)

ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('MonetaryValue')
ax.legend()

plt.show()


# --- DBSCAN Segmentation Cluster Size ---

# In[42]:


DBSCAN_clust_sizes= customers_normalized_df.groupby('Labels').size().to_frame()
DBSCAN_clust_sizes.columns= ["DBSCAN_size"]
DBSCAN_clust_sizes


# In[43]:


labels1= db.labels_
from sklearn import metrics
print(metrics.silhouette_score(customers_normalized_df, labels1))


# In[44]:


customers["Cluster"]= db.labels_
customers.groupby('Cluster').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'MonetaryValue':['mean', 'count']}).round(2)


# In[45]:


db.labels_.shape
customers["Cluster"]= db.labels_
customers.groupby('Cluster').agg({
                                  'Recency':'mean',
                                  'Frequency':'mean',
                                  'MonetaryValue':['mean', 'count']}).round(2)

f, ax= plt.subplots(figsize=(25, 5))
ax= sns.countplot(x= "Cluster", data= customers)
customers.groupby(['Cluster']).count()


# In[46]:


df_normalized = pd.DataFrame(customers_normalized_df, columns= ['Recency', 'Frequency', 'MonetaryValue'])
df_normalized['ID']= customers.index
df_normalized['Cluster']= db.labels_

df_nor_melt = pd.melt(df_normalized.reset_index(),
                      id_vars= ['ID', 'Cluster'],
                      value_vars=['Recency', 'Frequency', 'MonetaryValue'],
                      var_name= 'Attribute',
                      value_name= 'Value')
df_nor_melt.head()

sns.lineplot('Attribute', 'Value', hue= 'Cluster', data= df_nor_melt)


# ----------------------------------------------------------------------------------------------

# In[47]:


customers["Cluster"]= db.labels_
rfm_df= customers.groupby('Cluster').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'MonetaryValue':['mean', 'count']}).round(2)
rfm_df


# In[48]:


plt.figure(figsize=(12,10))
plt.subplot(3, 1, 1); sns.distplot(rfm_df['Recency'])
plt.subplot(3, 1, 2); sns.distplot(rfm_df['Frequency'])
plt.subplot(3, 1, 3); sns.distplot(rfm_df['MonetaryValue'])

plt.show()


# In[49]:


rfm_df= pd.DataFrame({
    'Recency': [120.07, 137.77, 159.83, 93.54, 260.20, 197.91, 396.71, 100.37, 385.48, 94.86, 9.00, 60.88, 9.18],
    'Frequency': [5.58, 5.00, 4.00, 8.59, 2.00, 3.00, 1.00, 6.00, 1.00, 7.00, 1.00, 1.00, 9.09],
    'MonetaryValue': [18.05, 12.87, 10.43, 21.84, 5.15, 7.81, 2.00, 15.43, 3.61, 17.75, 2.00, 3.25, 23.91],
    'Cluster': [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'RFM': [''] * 13
})

def map_to_rfm(row):
    recency_bins= [-float('inf'), 50, 100, 200, 300, float('inf')]
    recency_labels= ['VH', 'H', 'M', 'L', 'VL']
    
    fm_bins= [-float('inf'), 2.5, 3.5, 5.5, 7, float('inf')]
    fm_labels= ['VL', 'L', 'M', 'H', 'VH']
    
    rfm_categories= [recency_labels[i] for i in pd.cut([row['Recency']], bins= recency_bins, labels= False)]
    
    rfm_categories.extend([fm_labels[i] for i in pd.cut([row['Frequency'], row['MonetaryValue']], bins= fm_bins, labels= False)])
    
    return ' '.join(rfm_categories)

rfm_df['RFM']= rfm_df.apply(map_to_rfm, axis= 1)

print(rfm_df)


# In[ ]:





# ---  Cluster Segmentation  --

# In[50]:


rfm_df = pd.merge(rfm_df, DBSCAN_clust_sizes, left_on='Cluster', right_index=True, how='left')
rfm_df = rfm_df.rename(columns={'DBSCAN_size': 'DBSCAN_size'})
rfm_df['DBSCAN_size'] = rfm_df['DBSCAN_size'].fillna(0)

def map_segments(cluster):
    if cluster in [2, 11]:
        return 'High-Value Customers'
    elif cluster in [0, 1]:
        return 'Regular Spenders'
    elif cluster in [4, 6, 8]:
        return 'Big Spenders'
    elif cluster in [3, 5, 7]:
        return 'Low-Engagement Customers'
    elif cluster in [9, 10]:
        return 'Inactive Customers'
    else:
        return 'Noise'
    
rfm_df['Segments by Spending'] = rfm_df['Cluster'].apply(map_segments)

segmentation_df = rfm_df.groupby('Segments by Spending')['DBSCAN_size'].sum().reset_index()
segmentation_df.columns = ['Segments by Spending', 'Total DBSCAN_size']
segmentation_df


# In[51]:


import squarify
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(13, 7)
squarify.plot(sizes=segmentation_df['Total DBSCAN_size'], 
              label=['Big Spenders',
                     'High-Value Customers',
                     'Inactive Customers',
                     'Low-Engagement Customers',
                     'Noise', 
                     'Regular Spenders'], alpha=0.7)
plt.title("RFM Segments",fontsize=20)
plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




