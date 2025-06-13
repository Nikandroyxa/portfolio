#!/usr/bin/env python
# coding: utf-8

# --- Setup & Load Data ---

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df_fake= pd.read_csv('C://Users//User//Desktop//jobs//Tony Blair//archive (10)//Fake.csv')
df_fake


# In[3]:


df_true= pd.read_csv('C://Users//User//Desktop//jobs//Tony Blair//archive (10)//True.csv')
df_true


# In[4]:


df_fake.info()


# In[5]:


df_true.info()


# In[ ]:





# --- Preparation & Labeling ---
# 
#     Prepare the dataset for binary classification:
#     - We assign a label of '0' to all fake articles and '1' to all real articles
#     - We concatenate the 'title' and 'text' fields into a new column called 'full' to use as our input feature
#     - We retain only the relevant columns: 'full', 'subject', and 'label
#     - Finally, we concatenate the fake and real datasets into a single DataFrame for unified processing
# 
#     This results in a dataset of 44,898 articles

# In[6]:


df_fake['label']= 0
df_true['label'] = 1


# In[7]:


df= pd.concat([df_fake, df_true], ignore_index= True)
df['full']= df['title'] + ' ' + df['text']
df= df[['full', 'subject', 'label']]


# In[8]:


df


# In[ ]:





# --- Text Cleansing ---
# 
#     Prepare the text for vectorization
# 
#     - Applied a function 'preprocessing_pipeline' which:
#         - Converts all text to lowercase
#         - Removes multiple punctuation marks like "!!!" and "???"
#         - Removes URLs (`http...`), hashtags (`#...`), and mentions (`@...`)
#         - Strips all punctuation and digits
#         - Normalizes excessive whitespace
# 
#     - Also, tested the function to confirm it works

# In[ ]:





# In[9]:


import re
import string

def preprocessing_pipeline(text):
    text= text.lower().strip()

    text= re.sub(r'\!+', '!', text)
    text= re.sub(r'\?+', '?', text)

    text= ' '.join(word for word in text.split() if not word.startswith(('@', '#', 'http')))

    text= text.translate(str.maketrans('', '', string.punctuation))

    text= re.sub(r'\d+', '', text)

    text= re.sub(r'\s+', ' ', text)

    return text.strip()


# In[10]:


preprocessing_pipeline("Breaking!!! Check this out: https://fakeurl.com/news @user123 #breaking123 100% True???")


# In[11]:


df['full']= df['full'].apply(preprocessing_pipeline)


# In[12]:


df['full']


# In[ ]:





# In[ ]:





# --- Step 1 - Baseline ---

# In[ ]:





# In[ ]:





# --- Split - Train/Test Sets ---
# 
#     Splited the dataset into training set (80%) and test set (20%), 
#     stratifying on the 'label' column to ensure class balance in both sets
# 
#     Also, extracted the 'subject' column using: subj = df['subject']
#     
#     *This captures the domain or topic category of each article (e.g., "Politics", "News", "Middle-east", "WorldNews")
#      We store it separately as subj_train and subj_test so that we can later:
# 
#      - Evaluate model performance per domain
#      - Identify if the model performs better on certain topics
#      - Use it as an additional feature in the final improvement step

# In[ ]:





# In[13]:


from sklearn.model_selection import train_test_split

X= df['full']
y= df['label']

subj= df['subject']

X_train, X_test, y_train, y_test, subj_train, subj_test= train_test_split(
    X, y, subj, test_size= 0.2, stratify= y, random_state= 42)


# In[ ]:





# --- TF-IDF Vectorization (convert text to numeric features) ---
# 
#     To convert the text data into numerical format for machine learning, 
#     we applied TF-IDF (Term Frequency–Inverse Document Frequency) vectorization 
# 
#     Used the following parameters:
#         - 'stop_words= 'english'': removes common words like "the", "and", etc.
#         - 'max_df=0.8': ignores words that appear in more than 80% of documents
#         
#     Result:
#         - Training set: (35,918 samples, 185,039 features)
#         - Test set: (8,980 samples, 185,039 features)
# 
#     Also, checked the top TF-IDF weighted terms in the first few training documents 
#     and visualized the distribution of document lengths.

# In[ ]:





# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf= TfidfVectorizer(stop_words= 'english', max_df= 0.8)

X_train_vec= tfidf.fit_transform(X_train)
X_test_vec= tfidf.transform(X_test)


# In[15]:


print(f"TF-IDF Train Matrix Shape: {X_train_vec.shape}")
print(f"TF-IDF Test Matrix Shape: {X_test_vec.shape}")


# In[16]:


tfidf.get_feature_names_out()[:20]


# In[17]:


tfidf_df= pd.DataFrame(X_train_vec[:5].toarray(), columns= tfidf.get_feature_names_out())
tfidf_df.T.sort_values(by= 0, ascending= False).head(10)


# In[18]:


df['doc_length']= df['full'].apply(lambda x: len(x.split()))
sns.histplot(df['doc_length'], bins= 50)
plt.title("Distribution of Document Lengths (in words)")
plt.show()


# In[ ]:





# --- LogisticRegression ---
# 
#     Begun with LR(Logistic Regression)
# 
#     Results - Evaluation Metrics:
#         - Accuracy: 98.76%
#             This shows that 99% of articles were classified correctly as either fake or real
#             
#         - Precision: 98.40%
#             Out of all articles predicted as real, 98.4% were actually real.  
#             This is important when false positives (e.g., labeling fake news as real) are costly
#             
#         - Recall: 99%
#             Out of all actual real articles, 99% were correctly identified  
#             This reflects how well the model catches all real news
#             
#         - F1 Score: 99%
#             F1 balances both concerns and is especially useful when classes are imbalanced or 
#             when we care equally about precision and recall
#             
#         - AUC: 0.9878
#             Area Under the Receiver Operating Characteristic(ROC) Curve  
#             AUC near 1.0 indicates the model can almost perfectly separate fake from real news 
#             across all classification thresholds
# 
#     The confusion matrix shows strong performance on both fake and real classes, 
#     with a small misclassification.
# 
#     These results show us that a basic linear classifier performs exceptionally well on this dataset,
#     due to strong lexical differences between fake and real news articles.

# In[ ]:





# In[19]:


from sklearn.linear_model import LogisticRegression
lr= LogisticRegression(max_iter= 1000)
lr.fit(X_train_vec, y_train)

y_pred= lr.predict(X_test_vec)


# In[20]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, roc_auc_score, accuracy_score 

cm= confusion_matrix(y_test, y_pred, labels= lr.classes_)
print(classification_report(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))
print('AUC:', roc_auc_score(y_test, y_pred))
dcm_lr= ConfusionMatrixDisplay(cm, display_labels= lr.classes_)
dcm_lr.plot()


# In[ ]:





# --- Evaluation by Domain (subject) ---
# 
#     To understand how the model performs across different news domains, 
#     we evaluated accuracy, precision, recall, and F1-score separately for each value 
#     of the 'subject' field (e.g., "Politics", "News", "WorldNews")
# 
#     Grouped predictions by domain using a DataFrame that combined:
#         - 'subject' (domain)
#         - 'true' labels
#         - 'pred' labels
# 
#     During stratified train/test splitting, 
#     some domains such as PoliticsNews, WorldNews, and Left-News 
#     contain only one class(all real or all fake) in the test set
# 
#     Result:
#         - The model cannot predict the missing class
#         - Metrics like precision, recall, and F1 for that class are undefined and default to 0.0
#         - This probably is not a model issue, but a reflection of class imbalance within specific domains
# 
# 
#     Despite the class imbalance in some domains, the model maintained very high performance 
#     wherever both classes were present
#     This means that while the model generalizes well, domain diversity and label balance
#     should be considered in future dataset preparation.

# In[ ]:





# In[21]:


df_eval= pd.DataFrame({'subject': subj_test,
                       'true': y_test,
                       'pred': y_pred
                      })

from sklearn.metrics import classification_report

for domain in df_eval['subject'].unique():
    print(f"\n--- {domain.upper()} ---")
    d= df_eval[df_eval['subject']== domain]
    print(classification_report(d['true'], d['pred'], zero_division= 0))


# In[ ]:





# --- Feature Importance (Top Words) ---
# 
#     To understand how LR model makes decisions, extracted and visualized 
#     the most influential words based on the model's learned coefficients.
# 
#     Each word's weight (positive or negative) indicates how strongly 
#     it contributes to predicting an article as real or fake:
#             - Positive weights -> push the prediction toward real news
#             - Negative weights -> push the prediction toward fake news
# 
#     Visualized the top 20 most influential features and
#     sorted by the absolute value of their coefficients.
# 
#     - Words like "reuters", "said", and weekday names(e.g., wednesday, tuesday) are strongly 
#       associated with real news
#       
#     - Words like "video", "just", "hillary", and "gop" are 
#       associated with fake news

# In[ ]:





# In[22]:


feat_names= tfidf.get_feature_names_out()
coef= lr.coef_[0]


# In[23]:


feat_df= pd.DataFrame({'feature': feat_names,
                           'weight': coef
                          })

top_feat= feat_df.reindex(np.argsort(np.abs(feat_df['weight']))[::-1]).head(20)


# In[24]:


plt.figure(figsize= (10, 6))
sns.barplot(data= top_feat, x= 'weight', y= 'feature')
plt.title('Top 20 Most Influential Words')
plt.tight_layout()
plt.show()


# In[25]:


print("Top 20 Most Influential Words (Ordered by Importance):\n")

for i, row in top_feat.iterrows():
    print(f"{i+1}. {row['feature']} (weight= {row['weight']:.4f})")


# In[26]:


top_words= top_feat['feature'].tolist()
print(top_words)


# In[ ]:





# --- Top Predictors of Real News (positive weights) ----

# In[27]:


top_feat= top_feat.reset_index(drop= True)

top_real= top_feat[top_feat['weight'] > 0].sort_values(by= 'weight', ascending= False)

print("Words that strongly indicate **Real** news:\n")
for i, row in top_real.iterrows():
    print(f"{row['feature']} (weight= {row['weight']:.4f})")


# In[ ]:





# --- Top Predictors of Fake News (negative weights) ---

# In[28]:


top_fake= top_feat[top_feat['weight'] < 0].sort_values(by= 'weight')

print("\nWords that strongly indicate **Fake** news:\n")
for i, row in top_fake.iterrows():
    print(f"{row['feature']} (weight= {row['weight']:.4f})")


# In[ ]:





# In[ ]:





# --- Step 2 - Improvement ---

# In[ ]:





# In[ ]:





# --- Improvement 1 – TF-IDF with Bigrams ---
# 
#     We tried to enhanced the TF-IDF vectorizer by including bigrams('ngram_range=(1, 2)'), 
#     which allows the model to consider two-word phrases like:
#         - "donald trump"
#         - "white house"
#         - "fake news"
# 
#     Results - Evaluation(LR with Bigrams):
# 
#         - Accuracy (98.71%):  
#               The overall proportion of correctly predicted articles nearly 99% of predictions were correct
# 
#         - Precision (98.22%):  
#               Of all the articles predicted as real, 98.22% were actually real 
#   
#         - Recall (99%):  
#               Out of all actual real news articles, 99% were correctly identified  
#               This shows how well the model captures true positives
# 
#         - F1 Score (99%):  
#               A high F1 score indicates strong, consistent performance across both classes
# 
#         - AUC (0.9872):  
#               AUC close to 1 means near-perfect separability
# 
# 
#     While bigrams help capture multi-word expressions that may signal credibility or bias, 
#     the overall performance remained close to the unigram-based baseline
#     This indicates that the dataset is already highly separable using unigrams alone

# In[ ]:





# In[29]:


tfidf_bigram= TfidfVectorizer(stop_words= 'english',
                              max_df= 0.8,
                              ngram_range= (1, 2)
                             )

X_train_vec_bigram= tfidf_bigram.fit_transform(X_train)
X_test_vec_bigram= tfidf_bigram.transform(X_test)


# In[30]:


lr_bigram= LogisticRegression(max_iter= 1000)
lr_bigram.fit(X_train_vec_bigram, y_train)

y_pred_bigram= lr_bigram.predict(X_test_vec_bigram)

cm_bigram= confusion_matrix(y_test, y_pred_bigram, labels= lr_bigram.classes_)

print("Classification Report (LogReg with Bigrams):")
print(classification_report(y_test, y_pred_bigram))
print('Precision:', precision_score(y_test, y_pred_bigram))
print('Accuracy:', accuracy_score(y_test, y_pred_bigram))
print('AUC:', roc_auc_score(y_test, y_pred_bigram))

disp_bigram= ConfusionMatrixDisplay(cm_bigram, display_labels= lr_bigram.classes_)
disp_bigram.plot()


# In[ ]:





# ---  Improvement 2 – XGBoost ---
# 
#     We trained an XGBoost classifier, a powerful tree-based ensemble model known 
#     for its robustness, high performance, and ability to handle 
#     sparse data well — such as our TF-IDF matrix
# 
#     XGBoost is a non-linear model, which allows it to capture complex relationships 
#     in the data that linear models like Logistic Regression may miss
# 
#     Results - Evaluation (XGBoost on Unigram TF-IDF):
# 
#         - Accuracy (99.78%):  
#               Nearly all articles were correctly classified, showing a strong model fit
# 
#         - Precision (99.79%):  
#               Of all predicted real news articles, 99.79% were correct, indicating very low false positives
# 
#         - Recall (100%):  
#               The model successfully identified all real news articles in the test set meaning zero false negatives
# 
#         - F1 Score (1.00):  
#               Perfect balance between precision and recall
# 
#         - AUC (0.9978):  
#               The ROC-AUC score shows near-perfect class separability across thresholds
# 
# 
#     XGBoost significantly outperformed both Logistic Regression models
# 
#     This shows us that non-linear, tree-based models are highly effective
#     in detecting fake news, even when working with high-dimensional, sparse TF-IDF features

# In[ ]:





# In[31]:


from xgboost import XGBClassifier

xgb= XGBClassifier(use_label_encoder= False, eval_metric= 'logloss', random_state= 42)
xgb.fit(X_train_vec, y_train)

y_pred_xgb= xgb.predict(X_test_vec)

cm_xgb= confusion_matrix(y_test, y_pred_xgb, labels= xgb.classes_)

print("Classification Report (XGBoost):")
print(classification_report(y_test, y_pred_xgb))
print('Precision:', precision_score(y_test, y_pred_xgb))
print('Accuracy:', accuracy_score(y_test, y_pred_xgb))
print('AUC:', roc_auc_score(y_test, y_pred_xgb))

disp_xgb= ConfusionMatrixDisplay(cm_xgb, display_labels= xgb.classes_)
disp_xgb.plot()


# In[ ]:





# --- Final Improvement ---
# 
#     Finally we tried to enhanced our feature set by including domain metadata ('subject' column) 
#     with the TF-IDF text features
# 
#     We used:
#         - One-hot encoding for the 'subject' field (e.g., Politics, WorldNews, US_News)
#         - Combined it with the sparse TF-IDF matrix using 'scipy.sparse.hstack'
# 
#     Then we trained a LR model on the combined feature matrix
# 
#     Results - Evaluation (LogReg + TF-IDF + 'subject'):
# 
#         - Accuracy (100%):  
#               Every article in the test set was correctly classified
# 
#         - Precision (1.0):  
#               All articles predicted as real were truly real
# 
#         - Recall (1.0):  
#               All real news articles were correctly captured — no false negatives
# 
#         - F1 Score (1.0):  
#               Perfect balance between precision and recall.
# 
#         - AUC (1.0):  
#               The model achieved perfect separability between fake and real news.
# 
#     What does that mean in real world:
# 
#         While these results are impressive, such perfect performance in 
#         real-world datasets is rare and may indicate:
#             - Data leakage or Overfitting 
#             - Strong correlation between the 'subject' and the label, allowing the model 
#                 to "shortcut" classification without relying solely on article content
# 
#     This doesn't invalidate the result, it emphasizes the importance of understanding 
#     feature interactions and validating models on truly unseen or diverse data
# 
# 
#     By combining textual and categorical metadata, we were able to build a highly accurate and interpretable model
#     This shows how integrating multiple data sources can significantly enhance model performance in NLP tasks

# In[ ]:





# In[34]:


from sklearn.preprocessing import OneHotEncoder

enc= OneHotEncoder(handle_unknown= 'ignore', sparse= True)
subj_train_enc= enc.fit_transform(subj_train.values.reshape(-1, 1))
subj_test_enc= enc.transform(subj_test.values.reshape(-1, 1))


# In[35]:


from scipy.sparse import hstack

X_train_comb= hstack([X_train_vec, subj_train_enc])
X_test_comb= hstack([X_test_vec, subj_test_enc])

lr_comb= LogisticRegression(max_iter= 1000)
lr_comb.fit(X_train_comb, y_train)

y_pred_comb= lr_comb.predict(X_test_comb)

cm_comb= confusion_matrix(y_test, y_pred_comb, labels=lr_comb.classes_)

print("Classification Report (LogReg + Subject):")
print(classification_report(y_test, y_pred_comb))
print('Precision:', precision_score(y_test, y_pred_comb))
print('Accuracy:', accuracy_score(y_test, y_pred_comb))
print('AUC:', roc_auc_score(y_test, y_pred_comb))

disp_comb= ConfusionMatrixDisplay(cm_comb, display_labels= lr_comb.classes_)
disp_comb.plot()


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




