#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df= pd.read_csv("C://Users//User//Downloads//archive (9)//heart_disease_uci.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.describe().transpose()


# In[5]:


df.isnull().sum()


# In[6]:


dfq= df[['age', 'sex', 'cp', 'num']]
dfq


# In[7]:


dfq['sex']= dfq['sex'].str.lower()
dfq['cp']= dfq['cp'].str.lower()


# In[8]:


plt.hist(dfq['age'])


# In[9]:


def generalize_age(age):
    if age <= 40:
        return '0-40'
    elif age <= 50:
        return '41-50'
    elif age <= 60:
        return '51-60'
    else:
        return '61+'

dfq['age_group']= dfq['age'].apply(generalize_age)


# In[10]:


def generalize_cp(cp):
    if 'angina' in cp:
        return 'angina'
    elif 'non-anginal' in cp:
        return 'non-anginal'
    else:
        return 'other'

dfq['cp_generalized'] = dfq['cp'].apply(generalize_cp)


# In[11]:


def generalize_num(num):
    if num== 0:
        return 0
    else:
        return 1
    
dfq['num_gen']= dfq['num'].apply(generalize_num)


# In[12]:


dfq


# In[13]:


dfq['QID']= dfq['age_group']+ '_'+ dfq['cp_generalized']+ '_'+ dfq['sex']
dfq


# In[14]:


qid_count= dfq['QID'].value_counts().reset_index()
qid_count.columns= ['QID', 'QID_Count']
qid_count


# In[15]:


dfq= dfq.merge(qid_count, on= 'QID', how= 'left')
dfq


# In[16]:


dfq['QID_Count'].describe()


# In[17]:


dfq['QID_Count'].min()


# In[18]:


l_diversity= dfq.groupby('QID')['num_gen'].nunique().reset_index()
l_diversity.columns = ['QID', 'l_diversity']
dfq = dfq.merge(l_diversity, on='QID', how='left')


# In[19]:


dfq


# In[20]:


dfq['l_diversity'].value_counts().sort_index()


# In[21]:


dfq[dfq['l_diversity']==1]


# In[24]:


dfq= dfq[dfq['l_diversity'] > 1]
dfq


# In[26]:


overall_dist= dfq['num_gen'].value_counts(normalize=True).reset_index()
overall_dist.columns= ['num_gen', 'overall_freq']


# In[27]:


overall_dist


# In[28]:


group_dist= (
    dfq.groupby(['QID', 'num_gen'])
    .size()
    .groupby(level =0)
    .apply(lambda x: x / x.sum())
    .reset_index()
)
group_dist.columns= ['QID', 'num_gen', 'group_freq']


# In[29]:


group_dist


# In[30]:


t_closeness= group_dist.merge(overall_dist, on= 'num_gen')
t_closeness


# In[31]:


t_closeness['abs_diff']= abs(t_closeness['group_freq'] - t_closeness['overall_freq'])


# In[32]:


t_scores = t_closeness.groupby('QID')['abs_diff'].sum().reset_index()
t_scores.columns = ['QID', 't_closeness']


# In[33]:


t_scores


# In[34]:


t_scores.sort_values('t_closeness', ascending=False).plot.bar(x='QID', y='t_closeness', figsize=(12,6), color='skyblue')
plt.axhline(y=0.2, color='red', linestyle='--', label='Threshold (0.2)')
plt.title('t-Closeness by QID Group')
plt.ylabel('t-closeness score')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:





# In[35]:


risky_qids= t_scores[t_scores['t_closeness'] > 0.4]['QID']
dfq_safe= dfq[~dfq['QID'].isin(risky_qids)]
dfq_safe


# In[36]:


risky_qids


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




