#!/usr/bin/env python
# coding: utf-8

# --- Data Loading & Initial Exploration ---
#     
#     Importing the necessary libraries and load the two datasets
# 
#     df_a: Contains population, land area, and demographic data for each 5-digit ZIP Code Tabulation Area (ZCTA)
#     df_z: Contains geographical and state-level information for each ZIP code.
# 
#     info() function used to understand the structure, data types, completeness and 
#     helps to identify columns of interest and potential data quality issues before merging

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df_a= pd.read_csv('C://Users//User//Desktop//jobs//DataVant//Task//tasksheetdataa.csv')
df_a


# In[3]:


df_a.info()


# In[4]:


df_z= pd.read_csv('C://Users//User//Desktop//jobs//DataVant//Task//tasksheetdataz.csv')
df_z


# In[5]:


df_z.info()


# In[ ]:





# --- Data Preparation – ZIP Code Standardization & Merge ---
#     
#     We standardized the ZIP code columns in both dataframes:
#     df_a['zcta5'] and df_z['zip'] were converted to strings and filled to 5 digits using str.zfill(5)
#     This ensures consistency, for example, the ZIP code 356 becomes '00356'
#     
#     We merge the 2 datasets to 1 (df_merged) with inner join using the common column (zip code)
#     
#     The df_merged['arealand'] column in the dataset represents the land area of each ZIP code in square meters
#     and we converted it into square kilometers to be more understandable
# 
#     

# In[ ]:





# In[6]:


df_a['zcta5']= df_a['zcta5'].astype(str).str.zfill(5)
df_z['zip']= df_z['zip'].astype(str).str.zfill(5)


# In[7]:


df_a['zcta5'][0]


# In[8]:


df_z[df_z['zip'].str.len()!= 5]


# In[9]:


df_z[df_z['zip'].str.startswith('0')].head(5)


# In[10]:


df_merged = pd.merge(df_a, df_z, left_on= 'zcta5', right_on= 'zip', how= 'inner')
df_merged.shape


# In[11]:


df_merged.head(5)


# In[30]:


pd.set_option('display.max_columns', None)
df_merged[(df_merged['state']== 'Wyoming') & (df_merged['pop100']< 20000)]


# In[ ]:


28445	834	Alta	Wyoming

821	Wyoming	369	7698.493836	0.047931


# In[ ]:





# In[12]:


df_merged['area_km2']= df_merged['arealand']/ 1000000


# In[13]:


df_merged


# In[ ]:





# --- Data Cleaning — Handling Missing Values ---
# 
#     We explored the structure of df_merged and the columns 'community' and 'community_code' 
#     had almost all missing values (only 1 non-null entry), so they were dropped
#     
#     The columns 'state', 'stusab_y', 'county' and 'county_fips' had only 2 missing values, so either
#     we could delete those rows (as 2 out of 32974 rows is <0.01% of the dataset) or we could try to replace 
#     them with the actual as we did by finding columns which shares the same ZIP codes '96860' and '96863'

# In[ ]:





# In[14]:


df_merged.info()


# In[15]:


df_merged.isnull().sum()


# In[16]:


df_merged.drop(columns= ['community', 'community_code'], inplace= True)


# In[17]:


df_merged['stusab_x']


# In[18]:


df_merged['stusab_y']


# In[19]:


df_merged['state']


# In[20]:


df_merged['county']


# In[21]:


df_merged['county_fips']


# In[22]:


pd.set_option('display.max_columns', None)
df_merged[df_merged[['state', 'stusab_y', 'county']].isnull().any(axis=1)]


# In[23]:


df_merged[df_merged['zip'].isin(['96860', '96863'])]


# In[24]:


df_merged.loc[(df_merged['zip'].isin(['96860', '96863'])) & (df_merged['state'].isnull()),
             ['state', 'stusab_y', 'county', 'county_fips']]= ['Hawaii', 'HI', 'Honolulu', 3.0]


# In[25]:


df_merged[(df_merged['zip']== '96860') | (df_merged['zip']== '96863')]


# In[26]:


df_merged.info()


# In[27]:


df_merged.isnull().sum()


# In[ ]:





# --- Question 1 ---
# 
#     We grouped our data by state and counted the number of unique 5-digit ZIP codes per state

# In[ ]:





# In[28]:


most_zcodes= df_merged.groupby('state')['zip'].nunique().sort_values(ascending=False)
most_zcodes.head(5)


# In[29]:


print('The state with the most 5-digit ZIP codes is', most_zcodes.idxmax(), 'with', most_zcodes.max(), 'unique Zip codes')


# In[ ]:





# --- Question 2 ---

# --- Question 2.i ---
#     
#     To find the most easterly ZIP code in the U.S., we sorted our data by longitude in descending order as
#     the larger (less negative) the longitude, the closer to the eastern edge of the country it is 

# In[ ]:





# In[30]:


df_merged['longitude'].max()


# In[31]:


top_easterly= df_merged.sort_values(by= 'longitude', ascending= False)
top_easterly[['zip', 'city', 'state', 'longitude']].head(5)


# In[32]:


print(f"The 5-digit ZIP code which is the most easterly is {top_easterly.iloc[0]['zip']} and refers to {top_easterly.iloc[0]['city']}, {top_easterly.iloc[0]['state']}")


# In[ ]:





# --- Question 2.ii ---
#     
#     Since longitude gets more negative the further west you go, the minimum value of longitude 
#     gives us the western point
#     
#     We created a new dataframe (df_merged_noAlaska) without Alaska
#     Then checked in our intial df (df_merged) with Alaska to find the top_west_states see the second 
#     western state (Hawaii) to verify our output without Alaska later.

# In[ ]:





# In[33]:


df_merged_noAlaska= df_merged[df_merged['state']!= 'Alaska']


# --- With Alaska ---

# In[34]:


df_merged['longitude'].min()


# In[35]:


top_west= df_merged.sort_values(by= 'longitude')
top_west[['zip', 'city', 'state', 'longitude']].head(5)


# In[36]:


top_west_states= df_merged.groupby('state')['longitude'].min().sort_values()
top_west_states.head(5)


# --- Without Alaska ---

# In[37]:


df_merged_noAlaska['longitude'].min()


# In[38]:


top_westerly_noAlaska= df_merged_noAlaska.sort_values(by= 'longitude')
top_westerly_noAlaska[['zip', 'city', 'state', 'longitude']].head(5)


# In[39]:


print(f"The 5-digit ZIP code which is the most westerly, excluding Alaska, is {top_westerly_noAlaska.iloc[0]['zip']} and refers to {top_westerly_noAlaska.iloc[0]['city']}, {top_westerly_noAlaska.iloc[0]['state']}")


# In[ ]:





# --- Question 2.iii ---
#     
#     To find the most northern ZIP code, we look at the maximum latitude
#     Same as before we checked once with Alaska and then without

# In[ ]:





# --- With Alaska ---

# In[40]:


north= df_merged.sort_values(by= 'latitude')
north[['zip', 'city', 'state', 'latitude']]


# In[41]:


top_north_states= df_merged.groupby('state')['latitude'].max().sort_values(ascending= False)
top_north_states.head(5)


# In[42]:


top_northest= df_merged.sort_values(by= 'latitude', ascending= False)
top_northest[['zip', 'city', 'state', 'latitude']].head(5)


# In[43]:


print(f"The 5-digit ZIP code which is the most northerly is {top_northest.iloc[0]['zip']} and refers to {top_northest.iloc[0]['city']}, {top_northest.iloc[0]['state']}")


# --- Without Alaska ---

# In[44]:


top_northerly_noAlaska= df_merged_noAlaska.sort_values(by= 'latitude', ascending= False)
top_northerly_noAlaska[['zip', 'city', 'state', 'latitude']].head(5)


# In[45]:


print(f"The 5-digit ZIP code which is the most northerly, excluding Alaska, is {top_northerly_noAlaska.iloc[0]['zip']} and refers to {top_northerly_noAlaska.iloc[0]['city']}, {top_northerly_noAlaska.iloc[0]['state']}")


# In[ ]:





# --- Question 3 ---
#     
#     We calculated the population density for each 5-digit ZIP code by dividing 
#     the total population ('pop100') by the land area ('area_km2')
#     
#     The top 5 most densely populated ZIP codes were visualized using a bar chart, 
#     showing both the ZIP and the associated state

# In[ ]:





# In[46]:


df_merged['pop_density_km2']= round(df_merged['pop100'] / df_merged['area_km2'])
df_merged['pop_density_km2']


# In[47]:


df_merged['pop_density_km2'].max()


# In[48]:


top_pop_density= df_merged.sort_values(by= 'pop_density_km2', ascending= False)
top_pop_density[['zip', 'city', 'state', 'pop_density_km2']].head(5)


# In[49]:


print(f"The 5-digit ZIP code with the highest population density is {top_pop_density.iloc[0]['zip']}, located in {top_pop_density.iloc[0]['city']}, {top_pop_density.iloc[0]['state']}.")
print(f"It has a population density of {top_pop_density.iloc[0]['pop_density_km2']} people per square kilometer.")


# In[50]:


top5_density= top_pop_density.head(5).copy()
top5_density['label']= top5_density['zip'] + ' (' + top5_density['state'] + ')'

plt.figure(figsize=(10, 6))
sns.barplot(x= 'label', y= 'pop_density_km2', data= top5_density, palette= 'viridis')

plt.title('Top 5 Most Densely Populated ZIP Codes', fontsize= 14)
plt.xlabel('ZIP Code (State)')
plt.ylabel('Population Density (People/km²)')
plt.tight_layout()
plt.show()


# --- Question 4 ---

# --- Question 4.a ---
# 
#     Created a new column 'zip3c' by slicing the first three digits of the ZIP code
#     We grouped by that and counted the number of unique states per zip3c
#     Then we found how many states share the same 3-digit ZIP codes and which are they

# In[51]:


df_merged['zip3c']= df_merged['zip'].str[:3]
df_merged.head(5)


# In[52]:


zip3c_states= df_merged.groupby('zip3c')['state'].nunique().reset_index()
zip3c_states.columns= ['zip3c', 'num_states']
zip3c_states


# In[53]:


states_with_com_zip3c= zip3c_states[zip3c_states['num_states']> 1]
states_with_com_zip3c


# In[54]:


print(f"The 3-digit ZIP codes, which are common to more than one state is {states_with_com_zip3c.iloc[0]['zip3c']} and {states_with_com_zip3c.iloc[1]['zip3c']} ")


# In[55]:


df_merged[df_merged['zip3c'].isin(['063', '834'])].groupby('zip3c')['state'].unique()


# In[ ]:





# --- Question 4.b ---
# 
#     First created a filtered dataset (com_zip3c) containing only the entries 
#     where the 3-digit ZIP code (zip3c ('063' and '834')) is shared across more than one state
#     
#     Extracted unique combinations of the 3-digit ZIP code, city, and state to identify incongruous places
#     Removed the duplicates as we want to present the cities who belong in those states
#     Made a list with those cities

# In[ ]:





# In[56]:


com_zip3c= df_merged[df_merged['zip3c'].isin(states_with_com_zip3c['zip3c'])]
com_zip3c


# In[57]:


unique_com_zip3c= com_zip3c[['zip3c', 'city', 'state']].drop_duplicates()


# In[58]:


unique_com_zip3c= unique_com_zip3c.sort_values(by=['zip3c', 'state', 'city'])
unique_com_zip3c


# In[59]:


inc_places= unique_com_zip3c['city'].unique().tolist()
inc_places= ', '.join(sorted(inc_places))
print(inc_places)


# In[ ]:





# --- Question 5 ---
# 
#     Created a df (df_no_com_zip3c) exclude the places from Q4
#     Selected only the needed columns ('zip3c', 'state', 'pop100', 'area_km2')
#     Grouped and aggregated by summing the population and land area per ZIP3-state

# In[ ]:





# In[60]:


df_no_com_zip3c= df_merged[~df_merged['zip3c'].isin(states_with_com_zip3c['zip3c'])]
df_no_com_zip3c


# In[61]:


df_zip3_clean= df_no_com_zip3c[['zip3c', 'state', 'pop100', 'area_km2']]
df_zip3_clean


# In[62]:


zip3_grouped= df_zip3_clean.groupby(['zip3c', 'state']).agg(population= ('pop100', 'sum'), land_area_km2= ('area_km2', 'sum')).reset_index()
zip3_grouped


# In[ ]:





# --- Question 5.a ---
#     
#     I filtered the dataset using 10 <= population < 20000 to have
#     a population smaller than 20,000 (ignoring zeros and assuming single-digit counts as zero)   

# In[ ]:





# In[63]:


small_pop= zip3_grouped[(zip3_grouped['population']< 20000) & (zip3_grouped['population']>= 10)]
small_pop_sorted= small_pop.sort_values(by= 'population', ascending= False)
small_pop_sorted


# In[64]:


print(f"The 3-digit ZIP codes, which have a population smaller than 20,000 residents is: {small_pop.shape[0]}")


# In[ ]:





# --- Question 5.b ---
# 
#     Calculated the population density (people per km²) for all 3-digit ZIP codes
#     Because of the huge range of density values I used 3 plots to show population density/km2
#     
#     1) Log-Scaled Boxplot
#        We used log scare to capture the skewed distribution, 
#        especially due to extreme outliers like: New York,  District of Columbia and New Hampshire
#        
#     2) Distribution Plot (Raw Scale)
#        Gave us a general feel of how density values are distributed
#        
#     3) Distribution Plot (Log Scale)
#        To overcome skewness and emphasize the spread, we applied a logarithmic transformation to the x-axis
#        
#     Most of 3-digit ZIP codes in this population range are very sparsely populated,
#     indicating large wilderness areas with density under 6 people/km²
#     New York (zip3c: 102) is an extreme outlier due to its tiny area and built-up nature

# In[ ]:





# In[65]:


small_pop_sorted['pop_density_km2']= small_pop_sorted['population']/ small_pop_sorted['land_area_km2']


# In[66]:


small_pop_sorted


# In[67]:


small_pop_sorted.sort_values(by= 'pop_density_km2', ascending= False)


# In[68]:


plt.figure(figsize= (10, 6))
sns.boxplot(x= small_pop_sorted['pop_density_km2'])

plt.xscale('log')
plt.title('Population Density (Log Scale) of Small ZIP3 Regions')
plt.xlabel('Population Density (log scale, people/km²)')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[69]:


plt.figure(figsize=(10, 6))
sns.set(style = 'whitegrid')
sns.distplot(small_pop_sorted['pop_density_km2'])
plt.title('Distribution of Population Density – Small ZIP3s', fontsize = 20)
plt.xlabel('Population Density (People/km²)')
plt.ylabel('Count')


# In[70]:


plt.figure(figsize= (10, 6))
sns.histplot(data= small_pop_sorted, x= 'pop_density_km2', bins= 30, kde= True, log_scale= (True, False))

plt.title('Distribution of Population Density (log scale) – Small ZIP3s')
plt.xlabel('Population Density (People/km², log scale)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:





# --- Question 5.c ---
# 
#     Wyoming is the 3-digit ZIP code with the smallest population density
#     It is not a suprise as is well-known for its vast open landscapes (plains, mountains, national parks and ranches)
#     
#     !!! Fun Fact !!!!
#     
#     Wyoming's nicknamed the Equality State because it was the first state to grant women the right to vote 
#     and to have women serve on juries and hold public office with the first to elect a female governor:Nellie Tayloe Ross

# In[ ]:





# In[71]:


small_pop_sorted


# In[72]:


print(f"The 3-digit ZIP code with the smallest population density is in {small_pop_sorted.iloc[12]['state']} with population density: {small_pop_sorted.iloc[12]['pop_density_km2']}")


# In[ ]:





# --- Question 6 ---
# 
#     For the comparison of the distributions of population sizes for 5-digit 
#     and 3-digit ZIP codes we used df_merged and zip3_grouped dataframes, where in the
#     1st we used only the 5 most necessary columns ('zcta5', 'state', 'pop100', 'area_km2', 'pop_density_km2')
#     and created df_merged_pop
#     
#     We plot the distributions of population sepertly and then we compared them together in one plot

# In[ ]:





# --- Distribution of population sizes for 5-digit ---
#     
#     Shows a much wider spread and higher peak density values. There are extreme outliers, with population densities exceeding       10,000 people/km², indicating very dense urban areas like Manhattan (NYC) or Washington, D.C.
#     The distribution is right-skewed, with most ZIP codes having moderate to low density but a long tail toward high values.

# In[73]:


df_merged


# In[74]:


df_merged_pop= df_merged[['zcta5', 'state', 'pop100', 'area_km2', 'pop_density_km2']]
df_merged_pop


# In[75]:


plt.figure(figsize=(10, 6))
sns.set(style = 'whitegrid')
sns.distplot(df_merged_pop['pop_density_km2'])
plt.title('Distribution of Population Density – ZIP5s', fontsize = 20)
plt.xlabel('Population Density (People/km²)')
plt.ylabel('Count')


# In[76]:


clean_pop_density= df_merged_pop['pop_density_km2']
clean_pop_density= clean_pop_density.replace([np.inf, -np.inf], np.nan).dropna()
clean_pop_density= clean_pop_density[clean_pop_density > 0]

plt.figure(figsize= (10, 6))
sns.histplot(clean_pop_density, bins= 100, kde= True, log_scale= True, color= 'skyblue', edgecolor= 'black')

plt.title('Distribution of Population Density – 5-Digit ZIP Codes', fontsize= 16)
plt.xlabel('Population Density (People/km², log scale)', fontsize= 12)
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:





# --- Distribution of population sizes for 3-digit ---
# 
#     As expected, this distribution is more compact and aggregating ZIP5 codes into ZIP3 areas 
#     smooths out the extreme peaks and reduces variation. 
#     The highest densities are still visible, but much less frequent.

# In[77]:


zip3_grouped


# In[78]:


zip3_grouped['pop3_density_km2']= zip3_grouped['population']/ zip3_grouped['land_area_km2']
zip3_grouped


# In[79]:


plt.figure(figsize=(10, 6))
sns.set(style = 'whitegrid')
sns.distplot(zip3_grouped['pop3_density_km2'])
plt.title('Distribution of Population Density – ZIP3s', fontsize = 20)
plt.xlabel('Population Density (People/km²)')
plt.ylabel('Count')


# In[81]:


clean_pop3_density= zip3_grouped['pop3_density_km2']
clean_pop3_density= clean_pop3_density.replace([np.inf, -np.inf], np.nan).dropna()
clean_pop3_density= clean_pop3_density[clean_pop3_density > 0]

plt.figure(figsize= (10, 6))
sns.histplot(clean_pop3_density, bins= 100, kde= True, log_scale= True, color= 'salmon', edgecolor= 'black')

plt.title('Distribution of Population Density – 3-Digit ZIP Codes', fontsize= 16)
plt.xlabel('Population Density (People per km², log scale)', fontsize= 12)
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()


# --- Distribution of population sizes for 5-digit &  3-digit ---
# 
#     ZIP3 areas provide a broader, less granular view of population distribution, 
#     ZIP5 codes are better for identifying hyper-dense urban regions.

# In[87]:


clean_pop_density= df_merged_pop['pop_density_km2'].replace([np.inf, -np.inf], np.nan).dropna()
clean_pop_density= clean_pop_density[clean_pop_density > 0]

clean_pop3_density= zip3_grouped['pop3_density_km2'].replace([np.inf, -np.inf], np.nan).dropna()
clean_pop3_density= clean_pop3_density[clean_pop3_density > 0]

combined_density= pd.concat([
    pd.DataFrame({'Population Density': clean_pop_density, 'ZIP Level': 'ZIP5'}),
    pd.DataFrame({'Population Density': clean_pop3_density, 'ZIP Level': 'ZIP3'})
], ignore_index= True)

plt.figure(figsize= (12, 7))
sns.histplot(
    data= combined_density,
    x= 'Population Density',
    hue= 'ZIP Level',
    bins= 100,
    log_scale= True,
    kde= True,
    palette= ['skyblue', 'salmon'],
    edgecolor= 'black')

plt.title('Comparison of Population Density Distributions – ZIP3 vs ZIP5', fontsize= 16)
plt.xlabel('Population Density (People/km², log scale)', fontsize= 12)
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




