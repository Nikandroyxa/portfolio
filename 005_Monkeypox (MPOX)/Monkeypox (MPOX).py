#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df= pd.read_csv('C://Users//User//Desktop//MSc Westminster//Data Mining and Machine Learning//CourseWork//Monkeypox Coursework Dataset.csv')


# In[3]:


df= df.drop(['Test ID', 'Systemic Illness', 'Penile Oedema', 'Solitary Lesion', 'Home ownership', 'Month of Birth', 'Health Insurance', 'Red blood cells count', 'White blood cells count'], axis= 1)


# In[4]:


df= df.replace(' ', np.nan)


# In[5]:


df= df.dropna()


# In[6]:


def oral(value):
    if value == 'YES':
        return 1
    elif value == 'No':
        return 0
    else:
        return value


# In[7]:


df['Oral Lesions']= df['Oral Lesions'].apply(oral)
df['Oral Lesions']= df['Oral Lesions'].astype(int)


# In[8]:


def MPOX(v):
    if v== 'Negative':
        return 0
    elif v== 'Positive':
        return 1
    else:
        return v


# In[9]:


df['MPOX PCR Result']= df['MPOX PCR Result'].apply(MPOX)
df['MPOX PCR Result']= df['MPOX PCR Result'].astype(int)


# In[10]:


df.iat[291, 6]= 20
df['Age']= pd.to_numeric(df['Age'])


# In[11]:


df['Age']= df['Age'].replace([-23, 0, 150, 181], np.nan)
df= df.dropna()


# In[12]:


df['Encoded Systemic Illness']= df['Encoded Systemic Illness'].astype(int)
df['Rectal Pain']= df['Rectal Pain'].astype(int)
df['Swollen Tonsils']= df['Swollen Tonsils'].astype(int)
df['HIV Infection']= df['HIV Infection'].astype(int)
df['Sexually Transmitted Infection']= df['Sexually Transmitted Infection'].astype(int)
df['Age']= df['Age'].astype(int)


# In[13]:


df.info()


# In[14]:


X= df.drop('MPOX PCR Result', axis= 1)
y= df['MPOX PCR Result']


# In[ ]:





#               ~~ LogisticRegression ~~

# In[ ]:





# In[15]:


from sklearn.preprocessing import StandardScaler
ss= StandardScaler()
X1= ss.fit_transform(X)


# In[16]:


from sklearn.model_selection import train_test_split
X1_train, X1_test, y_train, y_test= train_test_split(X1, y, random_state= 5, test_size= 0.3, stratify= y)


# In[17]:


from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(X1_train, y_train)


# In[18]:


lr_intercept= lr.intercept_
lr_slope= lr.coef_
print('Intercept: ',lr_intercept)
print('Slope: ',lr_slope)


# In[19]:


y_lr_tr= lr.predict(X1_train)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
cm_lr_tr= confusion_matrix(y_train, y_lr_tr, labels= lr.classes_)
print(cm_lr_tr)
print('AUC:', roc_auc_score(y_train, y_lr_tr))
print(classification_report(y_train, y_lr_tr))
dcm_lr_tr= ConfusionMatrixDisplay(cm_lr_tr, display_labels= lr.classes_)
dcm_lr_tr.plot()


# In[20]:


y_lr= lr.predict(X1_test)
cm_lr= confusion_matrix(y_test, y_lr, labels= lr.classes_)
dcm_lr= ConfusionMatrixDisplay(cm_lr, display_labels= lr.classes_)
cr_lr= classification_report(y_test, y_lr)
auc_lr= roc_auc_score(y_test, y_lr)
print(cm_lr)
print('AUC:',auc_lr)
print(cr_lr)
dcm_lr.plot()


# In[ ]:





#               ~~ DecisionTreeClassifier ~~

# In[ ]:





# In[21]:


from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier()
dt.fit(X1_train, y_train)


# In[22]:


y_dt= dt.predict(X1_test)
cm_dt= confusion_matrix(y_test, y_dt, labels= dt.classes_)
auc_dt= roc_auc_score(y_test, y_dt)
cr_dt= classification_report(y_test, y_dt)
dcm_dt= ConfusionMatrixDisplay(cm_dt, display_labels= dt.classes_)
print(cm_dt)
print('AUC:', auc_dt)
print(cr_dt)
dcm_dt.plot()


# In[ ]:





#               ~~ DecisionTreeClassifier - PreProuning ~~

# In[ ]:





# In[23]:


from sklearn.tree import DecisionTreeClassifier
dt_pr= DecisionTreeClassifier(criterion='entropy', max_depth= 3)
dt_pr.fit(X1_train, y_train)


# In[24]:


y_dt_pr= dt_pr.predict(X1_test)
cm_dt_pr= confusion_matrix(y_test, y_dt_pr, labels= dt_pr.classes_)
print(cm_dt_pr)
print('AUC: ', roc_auc_score(y_test, y_dt_pr))
print(classification_report(y_test, y_dt_pr))
dcm_dt_pr= ConfusionMatrixDisplay(cm_dt_pr, display_labels= dt_pr.classes_)
dcm_dt_pr.plot()


# In[ ]:





#               ~~ KNeighborsClassifier ~~

# In[ ]:





# In[25]:


from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors= 3)
knn.fit(X1_train, y_train)


# In[26]:


y_knn= knn.predict(X1_test)
cm_knn= confusion_matrix(y_test, y_knn, labels= knn.classes_)
print(cm_knn)
print('AUC:', roc_auc_score(y_test, y_knn))
print(classification_report(y_test, y_knn))
dcm_knn= ConfusionMatrixDisplay(cm_knn, display_labels= knn.classes_)
dcm_knn.plot()


# In[ ]:





# In[27]:


error= []
for i in range(1,25):
    knn1= KNeighborsClassifier(n_neighbors= i)
    knn1.fit(X1_train, y_train)
    y_knn1= knn1.predict(X1_test)
    error.append(np.mean(y_knn1!= y_test))


# In[28]:


plt.figure(figsize= (12, 6))
plt.plot(range(1, 25), error, color= 'red', linestyle= 'dashed', marker= 'o', markerfacecolor= 'blue', markersize= 10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# In[29]:


knn23= KNeighborsClassifier(n_neighbors= 23)
knn23.fit(X1_train, y_train)


# In[30]:


y_knn23_tr= knn23.predict(X1_train)
cm_knn23_tr= confusion_matrix(y_train, y_knn23_tr, labels= knn23.classes_)
print(cm_knn23_tr)
print('AUC:', roc_auc_score(y_train, y_knn23_tr))
print(classification_report(y_train, y_knn23_tr))
dcm_knn23_tr= ConfusionMatrixDisplay(cm_knn23_tr, display_labels= knn23.classes_)
dcm_knn23_tr.plot()


# In[31]:


y_knn23= knn23.predict(X1_test)
cm_knn23= confusion_matrix(y_test, y_knn23, labels= knn23.classes_)
print(cm_knn23)
print('AUC:', roc_auc_score(y_test, y_knn23))
print(classification_report(y_test, y_knn23))
dcm_knn23= ConfusionMatrixDisplay(cm_knn23, display_labels= knn23.classes_)
dcm_knn23.plot()


# In[ ]:





#               ~~ KNeighborsClassifier GSCV ~~

# In[ ]:





# In[32]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
knn_gs= KNeighborsClassifier()
grid_param= {'n_neighbors': np.arange(1,25)}
gscv_knn= GridSearchCV(knn_gs, grid_param, cv= 5)


# In[33]:


gscv_knn.fit(X1, y)


# In[34]:


print(gscv_knn.best_params_)
print(gscv_knn.best_score_)


# In[ ]:





#               ~~ GaussianNB ~~

# In[ ]:





# In[35]:


from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()
nb.fit(X1_train, y_train)


# In[36]:


y_nb= nb.predict(X1_test)
cm_nb= confusion_matrix(y_test, y_nb, labels= nb.classes_)
print(cm_nb)
print('AUC:', roc_auc_score(y_test, y_nb))
print(classification_report(y_test, y_nb))
dcm_nb= ConfusionMatrixDisplay(cm_nb, display_labels= nb.classes_)
dcm_nb.plot()


# In[ ]:





#               ~~ SVM(RBF) ~~

# In[ ]:





# In[37]:


from sklearn.svm import SVC
svc= SVC(kernel= 'rbf', gamma= 1)
svc.fit(X1_train, y_train)


# In[38]:


y_svc= svc.predict(X1_test)
cm_svc= confusion_matrix(y_test, y_svc, labels= svc.classes_)
print(cm_svc)
print('AUC:', roc_auc_score(y_test, y_svc))
print(classification_report(y_test, y_svc))
dcm_svc= ConfusionMatrixDisplay(cm_svc, display_labels= svc.classes_)
dcm_svc.plot()


# In[ ]:





#           ~~ SVM(RBF) -  GridSearchCV ~~

# In[ ]:





# In[39]:


from sklearn.model_selection import GridSearchCV
param_grid= {'C':[0.1,1,10,100], 
            }
gscv_svc = GridSearchCV(SVC(kernel= 'rbf'), param_grid, refit = True, verbose=3)
gscv_svc.fit(X1_train, y_train)


# In[40]:


print(gscv_svc.best_estimator_)


# In[41]:


grid_scores = gscv_svc.cv_results_


# In[42]:


y_gscv_svc= gscv_svc.predict(X1_test)
cm_gscv_svc= confusion_matrix(y_test, y_gscv_svc, labels= gscv_svc.classes_)
print(cm_gscv_svc)
print('AUC:', roc_auc_score(y_test, y_gscv_svc))
print(classification_report(y_test, y_gscv_svc))
dcm_gscv_svc= ConfusionMatrixDisplay(cm_gscv_svc, display_labels= gscv_svc.classes_)
dcm_gscv_svc.plot()


# In[ ]:





# In[43]:


from sklearn.svm import SVC
svc_best= SVC(kernel='rbf', C=100, gamma=0.1)
svc_best.fit(X1_train, y_train)


# In[44]:


y_svc_best= svc_best.predict(X1_test)
cm_svc_best= confusion_matrix(y_test, y_svc_best, labels= svc_best.classes_)
print(cm_svc_best)
print('AUC:', roc_auc_score(y_test, y_svc_best))
print(classification_report(y_test, y_svc_best))
dcm_svc_best= ConfusionMatrixDisplay(cm_svc_best, display_labels= svc_best.classes_)
dcm_svc_best.plot()


# In[ ]:





# In[ ]:





#       ~~ VotingClassifier - Soft ~~

# In[ ]:





# In[ ]:





# In[46]:


from sklearn.ensemble import VotingClassifier
base_learners= [('Knn23', knn23), ('lr', lr)]
ens_s= VotingClassifier(base_learners, voting= 'soft')
ens_s.fit(X1_train, y_train)


# In[47]:


y_ens_s= ens_s.predict(X1_test)


# In[48]:


cm_ens_s= confusion_matrix(y_test, y_ens_s, labels= ens_s.classes_)
print(cm_ens_s)
print('AUC:', roc_auc_score(y_test, y_ens_s))
print(classification_report(y_test, y_ens_s))
dcm_ens_s= ConfusionMatrixDisplay(cm_ens_s, display_labels= ens_s.classes_)
dcm_ens_s.plot()


# In[ ]:





#   ~~ VotingClassifier - Hard ~

# In[ ]:





# In[49]:


from sklearn.ensemble import VotingClassifier
base_learners= [('Knn23', knn23), ('lr', lr)]
ens_h= VotingClassifier(base_learners, voting= 'hard')
ens_h.fit(X1_train, y_train)


# In[50]:


y_ens_h= ens_h.predict(X1_test)
cm_ens_h= confusion_matrix(y_test, y_ens_h, labels= ens_h.classes_)
print(cm_ens_h)
print('AUC:', roc_auc_score(y_test, y_ens_h))
print(classification_report(y_test, y_ens_h))
dcm_ens_h= ConfusionMatrixDisplay(cm_ens_h, display_labels= ens_h.classes_)
dcm_ens_h.plot()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




