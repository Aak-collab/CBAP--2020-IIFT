#!/usr/bin/env python
# coding: utf-8

# In[44]:


import nltk
import nltk.corpus
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer


# In[2]:


nltk.download('wordnet')


# In[16]:


data = pd.read_excel("C:/Users/Aakash Predator/Downloads/TITANIC/ReviewsFileName.xlsx")


# In[3]:


data.head()


# In[4]:


data.Review.shape
stopwordslist = stopwords.words("english")
stopwordslist


# ##  Define a function which can perform the following functions:
# 
# Remove non-alphabets 
# Remove URLs
# Remove digits
# Remove stopwords
# Stem the texts using PorterStemmer
# Remove and replace “’”, “--”, “-”, “[”, “]” by “ ”.
#  3.   Create a list of 30 most frequently occurring words from cleaned reviews and write it to 'nlargest.txt'.
# 
# 4.   Create a train (67%) and test (33%) split with random state 42
# 
# 5.   Create a TF-IDF vector with the following parameter:
# 
# ngram_range = (1,2)
# max_df=0.3
# min_df=7
# 6.   Build a Random Forest Classifier [Preferably, perform step 5 and 6 together using Pipeline from sklearn]

# In[17]:



data.Review[0]   


# In[18]:


def text_cleaner(data):
    data= data.apply(lambda x:re.sub("[^a-zA-Z]+"," ",x) ) #removes digits and not alphabets
    data= data.apply(lambda x:re.sub(r"http\S+", "",x) )  #removes urls
    data= data.apply(lambda x: word_tokenize(x))#tokenize data review
    def remove_stopwords(text):
        words = [w for w in text if w not in stopwords.words('english')]
        return words
    data = data.apply(lambda x : remove_stopwords(x))#removing stopwords
    ps = PorterStemmer()
    data = data.apply(lambda x:[ps.stem(w) for w in x])#applying porter stemmer
    
    return data
    


# In[19]:


data_cleaned = text_cleaner(data.Review)


# In[21]:


data_cleaned[0]


# In[22]:


words_list = []
for i  in range(data_cleaned.shape[0]):
    for word in data_cleaned[i]:
        words_list.append(word)


# In[23]:


words_list


# In[27]:


from nltk.probability import FreqDist


# In[37]:


joined_text =" ".join(words_list)
joined_text


# In[38]:


fdist = FreqDist()
for word in words_list:
    fdist[word.lower()]+=1


# In[39]:


fdist.plot()


# In[40]:


fdist_top30=fdist.most_common(30)


# In[52]:


freq_words = []
for i in range(len(fdist_top30)):
    freq_words.append(fdist_top30[i][0])


# In[56]:


freq_words = str(freq_words)


# In[60]:


with open('nlargest.txt', 'w') as filehandle:
    for listitem in freq_words:
        filehandle.write('%s' % listitem)


# In[87]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

y=data.Sentiment
y.head()
# In[82]:


tf_vect = TfidfVectorizer(lowercase =False, stop_words ='english', min_df = 7, ngram_range=(1,2), max_df=0.3)
x_tf_vect = tf_vect.fit_transform(data.Review)
x_tf_vect.shape


# In[83]:


x_names = tf_vect.get_feature_names()
x_names


# In[84]:


X_tf_vect = pd.DataFrame(x_tf_vect.toarray(), columns=x_names)
X_tf_vect.head()


# In[85]:


x_train,x_test,y_train,y_test = train_test_split(X_count_vect, y, test_size=0.33, random_state=42)


# In[86]:


y_train


# In[88]:


rfc = RandomForestClassifier()


# In[91]:


model=rfc.fit(x_train,y_train)


# In[92]:


y_pred = model.predict(x_test)


# In[93]:


y_pred


# In[95]:


conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


# In[96]:


conf_matrix.tofile('cfmatrix.txt',sep=',')


# In[ ]:




