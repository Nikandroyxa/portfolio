#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# --- Reddit ---

# In[2]:


import praw
import pprint


# In[3]:


reddit= praw.Reddit(client_id= 'YAZp0-ohwkilZUSqq46lyA',
                     client_secret= 'DALnPYeSNU9ntRGDhaMgTvT4D77Mew',
                     user_agent= 'Comment Extraction')

def extract_comments(post_url):
    all_comments= []

    submission= reddit.submission(url= post_url)
    submission.comments.replace_more(limit= None)

    for comment in submission.comments.list():
        comment_data= {
            "timestamp": pd.to_datetime(comment.created_utc, unit= 's'),
            "text": comment.body
        }
        all_comments.append(comment_data)

        for reply in comment.replies.list():
            reply_data= {
                "timestamp": pd.to_datetime(reply.created_utc, unit= 's'),
                "text": reply.body
            }
            all_comments.append(reply_data)

    return all_comments

post_url= "https://www.reddit.com/r/tech/comments/ogtwtt/harleydavidson_launches_cheaper_livewire_electric/"
post_comments= extract_comments(post_url)

df = pd.DataFrame(post_comments)
df.drop_duplicates(inplace=True)
df.head(5)


# In[4]:


reddit= praw.Reddit(client_id= 'ccGYRVJbM0uuYJ36AL1Y8A',
                     client_secret= 'bZoc8ROK7UrN6NhAhlmXx0fPBhxAIw',
                     user_agent= 'Comment Extraction')

def extract_comments(post_url):
    all_comments= []

    submission= reddit.submission(url= post_url)
    submission.comments.replace_more(limit= None)

    for comment in submission.comments.list():
        comment_data= {
            "timestamp": pd.to_datetime(comment.created_utc, unit= 's'),
            "text": comment.body
        }
        all_comments.append(comment_data)

        for reply in comment.replies.list():
            reply_data= {
                "timestamp": pd.to_datetime(reply.created_utc, unit= 's'),
                "text": reply.body
            }
            all_comments.append(reply_data)

    return all_comments

post_url= "https://www.reddit.com/r/motorcycles/comments/vyzkyw/i_got_to_test_ride_the_harley_davidson_livewire/"
post_comments= extract_comments(post_url)

df1= pd.DataFrame(post_comments)
df1.drop_duplicates(inplace=True)
df1.head(5)


# In[5]:


reddit= praw.Reddit(client_id= '6TcuVwaS-2pXL9FIKCtAjg',
                     client_secret= 'a2dHhKuftXGAqbymX8EYHFXB3-LZtw',
                     user_agent= 'Comment Extraction')

def extract_comments(post_url):
    all_comments= []

    submission= reddit.submission(url= post_url)
    submission.comments.replace_more(limit= None)

    for comment in submission.comments.list():
        comment_data= {
            "timestamp": pd.to_datetime(comment.created_utc, unit= 's'),
            "text": comment.body
        }
        all_comments.append(comment_data)

        for reply in comment.replies.list():
            reply_data= {
                "timestamp": pd.to_datetime(reply.created_utc, unit= 's'),
                "text": reply.body
            }
            all_comments.append(reply_data)

    return all_comments

post_url= "https://www.reddit.com/r/motorcycles/comments/1cdtvu8/harley_cannot_sell_its_electric_livewires/"
post_comments= extract_comments(post_url)

df2= pd.DataFrame(post_comments)
df2.drop_duplicates(inplace=True)
df2.head(5)


# --- YouTube ---

# In[6]:


from googleapiclient.discovery import build


# In[7]:


api_key= 'AIzaSyCiVnIECfnX293eqec-HFLdgke1ux3rqi8'

def extract_comments_with_timestamp(video_ids):
    all_comments= []

    youtube= build('youtube', 'v3', developerKey=api_key)

    for video_id in video_ids:
        response= youtube.commentThreads().list(
            part= 'snippet',
            videoId= video_id,
            maxResults= 100
        ).execute()

        while response:
            for item in response['items']:
                comment_text= item['snippet']['topLevelComment']['snippet']['textOriginal']
                comment_timestamp= item['snippet']['topLevelComment']['snippet']['publishedAt']
                all_comments.append({'text': comment_text, 'timestamp': comment_timestamp})

                replies_response= youtube.comments().list(
                    part= 'snippet',
                    parentId= item['snippet']['topLevelComment']['id'],
                    maxResults= 100
                ).execute()

                for reply_item in replies_response['items']:
                    reply_text= reply_item['snippet']['textOriginal']
                    reply_timestamp= reply_item['snippet']['publishedAt']
                    all_comments.append({'text': reply_text, 'timestamp': reply_timestamp})

            if 'nextPageToken' in response:
                response= youtube.commentThreads().list(
                    part= 'snippet',
                    videoId= video_id,
                    pageToken= response['nextPageToken'],
                    maxResults= 100
                ).execute()
            else:
                break

    return all_comments

video_ids= ["iETp6m3wswQ", "jwTqRipq_Mo", "BwVdChfLD8I"]

video_comments_with_timestamp= extract_comments_with_timestamp(video_ids)

df3= pd.DataFrame(video_comments_with_timestamp)
df3= df3.drop_duplicates()
df3.head(5)


# In[8]:


df3['timestamp'] = pd.to_datetime(df3['timestamp'])
df3['timestamp'] = df3['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')


# In[9]:


df_final= pd.concat([df, df1, df2, df3], ignore_index= True)
df_final.head(5)


# --- Data Preprocessing ---

# In[10]:


def common_case(text):
    return text.lower()

def without_leading_trailing_whitespace(text):
    return text.strip()

import re
def no_multi_punctuation(text):
    pattern= r"\!+"
    text= re.sub(pattern, "!", text)
    pattern= r"\?+"
    text= re.sub(pattern, "?", text)
    return text

def no_recomments(text):
    keep= []
    for word in text.split():
        if not word.startswith("@"):
            keep.append(word)
    return ' '.join(keep)

def no_recomments1(text):
    keep= []
    for word in text.split():
        if not word.startswith("#"):
            keep.append(word)
    return ' '.join(keep)

def no_http_links(text):
    keep= []
    for word in text.split():
        if not word.startswith("http"):
            keep.append(word)
    return ' '.join(keep)


# In[11]:


def preprocessing_pipeline(text):
    text= common_case(text)
    text= without_leading_trailing_whitespace(text)
    text= no_multi_punctuation(text)
    text= no_recomments(text)
    text= no_recomments1(text)
    text= no_http_links(text)
    return text

preprocessing_pipeline("More and Less https://www.google.com @asdasdwe vbnvetet!!!")


# In[12]:


df_final['cleaned_text']= df_final['text'].apply(preprocessing_pipeline)
df_final


# In[13]:


def comment_len(row):
    return len(row["cleaned_text"])

df_final["len"] = df_final.apply(comment_len, axis=1)


# In[14]:


df_final["len"].describe()


# In[15]:


df_final= df_final[df_final["len"] >= 10].copy()
df_final.head(5)


# In[16]:


from langdetect import detect


# In[17]:


def language_code(row):
    try:
        return detect(row["cleaned_text"])
    except:
        return "Unknown"

df_final["lang"]= df_final.apply(language_code, axis=1)
df_final.head(5)


# In[18]:


df_final.to_csv('df_final.csv', index=False)


# In[ ]:





# --- EXPLORATORY ---

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[20]:


df_final['timestamp']= pd.to_datetime(df_final['timestamp'])
df_final.dtypes


# In[21]:


comments_per_day= df_final.groupby(df_final["timestamp"].dt.date).size()
comments_per_day.describe()


# In[22]:


plt.figure(figsize= (10, 6))
plt.plot(comments_per_day.index, comments_per_day.values, marker= 'o', linestyle= '-')
plt.title('Comments Per Day')
plt.xlabel('Date')
plt.ylabel('Number of Comments')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()


# In[23]:


top_5_dates_in_comments= comments_per_day.sort_values(ascending= False).head(5)
plt.figure(figsize= (8, 4))
plt.bar(top_5_dates_in_comments.index.astype(str), top_5_dates_in_comments.values, color= 'skyblue')
plt.title('Top 5 Days in Comments')
plt.xlabel('Date')
plt.ylabel('Number of Comments')
plt.xticks(rotation= 45)
plt.tight_layout()
plt.show()


# In[24]:


plt.figure(figsize=(8, 4))
sns.set(style = 'whitegrid')
sns.distplot(comments_per_day)
plt.title('Distribution of No of Comments per Day', fontsize = 20)
plt.xlabel('No of Comments')
plt.ylabel('Count')


# In[25]:


import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

def preprocess(row):
    text= row["text"]
    text= text.lower()
    keep= []
    for word in text.split():
        if word not in stopwords.words("english"):
            keep.append(word)
    return ' '.join(keep)

df_final["cleaned_text"]= df_final.apply(preprocess, axis= 1)
df_final.head(5)


# In[26]:


from collections import Counter

word_counter= Counter()
for row in df_final.to_dict("records"):
    word_counter.update(row["cleaned_text"].split())

df_term_freq= pd.DataFrame(word_counter.most_common(10))
df_term_freq.columns= ["term", "frequency"]
df_term_freq


# In[27]:


import gensim
import gensim.corpora as corpora
from pprint import pprint

documents= [comment.split() for comment in df_final["cleaned_text"]]
vocab= corpora.Dictionary(documents)
corpus= [vocab.doc2bow(text) for text in documents]

lda= gensim.models.LdaMulticore(corpus= corpus, id2word= vocab, num_topics= 5)
pprint(lda.print_topics())


# In[ ]:





# --- TEXT MINING ---

# In[28]:


df_lex = pd.read_csv("C://Users//User//Desktop//MSc Westminster//Web and Social Media Analytics//Week_12//2000.tsv", sep="\t", header=None)
df_lex.columns=["word","sentiment", "std.dev"]
df_lex.head(5)


# In[29]:


from matplotlib import pyplot as plt
df_lex['sentiment'].plot(kind='hist', bins= 20, title= 'sentiment')
plt.gca().spines[['top', 'right',]].set_visible(False)


# In[30]:


df_lex[["word", "sentiment"]].to_dict("records")


# In[31]:


mapping= {}
for row in df_lex[["word", "sentiment"]].to_dict("records"):
  mapping[row["word"]]= row["sentiment"]
mapping


# In[32]:


def sentiment_score(row):
    text= row["cleaned_text"].lower()
    score= 0
    n= 0
    for word in text.split():
        if word in mapping:
            n+= 1
            score+= mapping[word]
    if n> 0:
        return score/ n
    return 0

df_final["simple_sentiment"]= df_final.apply(sentiment_score, axis= 1)
df_final.head(5)


# In[33]:


df_final["simple_sentiment"].describe()


# In[34]:


df_final["simple_sentiment"].plot(kind= 'hist', bins=20, title= 'Simple Sentiment')
plt.gca().spines[['top', 'right',]].set_visible(False)


# In[35]:


nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia= SentimentIntensityAnalyzer()

def score_vader(row):
    text = row["cleaned_text"]
    return sia.polarity_scores(text)["compound"]

df_final["vader_sentiment"]= df_final.apply(score_vader, axis= 1)
df_final.head(5)


# In[36]:


df_final["vader_sentiment"].describe()


# In[37]:


plt.figure(figsize=(12,10))
plt.subplot(3, 1, 1); sns.distplot(df_final["simple_sentiment"])
plt.subplot(3, 1, 2); sns.distplot(df_final["vader_sentiment"])
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




