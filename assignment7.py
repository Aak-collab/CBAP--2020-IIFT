#!/usr/bin/env python
# coding: utf-8

# # Assignment 7 

# In[1]:


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import PCA


# In[4]:


with open('C:/Users/Aakash Predator/OneDrive/Desktop/Datasets/NewsArticles.json', 'r') as f:
    df = json.load(f)
    
    


# In[ ]:





# In[6]:


data = pd.DataFrame(df)
data


# In[44]:


#Deleting dupliate headlines(if any)
data[data['Preprocessed-Article'].duplicated(keep=False)].sort_values('Preprocessed-Article').head(8)


# In[45]:


data = data.drop_duplicates('Preprocessed-Article')
data


# In[8]:


documents = data['Preprocessed-Article']
vect = TfidfVectorizer(stop_words='english')
X1 = vect.fit_transform(documents)
model = KMeans(n_clusters=5,random_state=1)
model.fit(X1)
lst2 = []
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vect.get_feature_names()
for i in range(5):
    print('Cluster %d:' %i)
    for ind in order_centroids[i, :50]:
        print('%s'%terms[ind])
        lst2.append(' %s' % terms[ind])
print(lst2)
#calculate SSE for each k
sse = {}
for k in range(0,5):
    sse[k] =model.inertia_
    
round(sse[0],2 )   
s5 = lst2[49]
s5


# In[18]:


pca = PCA(n_components=100, random_state=1,svd_solver='full')
pca.fit(X1)
pca.transform(X1)


# In[29]:


punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)
desc = data['Preprocessed-Article'].values
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(desc)
X


# In[52]:


word_features = vectorizer.get_feature_names()
len(word_features)


# In[53]:


stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')
def tokenize(text):
    return [stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]


# In[66]:


#Vectorization with stop words(words irrelevant to the model), stemming and tokenizing
#ectorizer2 = TfidfVectorizer(stop_words=stop_words,tokenizer=tokenize)
#2 = vectorizer2.fit_transform(desc)
#ord_features2 = vectorizer2.get_feature_names()


# In[30]:


#K-means clustering
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)


# In[31]:


common_words = kmeans.cluster_centers_.argsort()[:,-1:-50:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(word_features[word] for word in centroid))


# In[65]:





# In[10]:


det = data['Vector']

vectors=pd.DataFrame(list(det))
vectors


# In[21]:


kmeans_model = KMeans(n_clusters = 5,random_state=1).fit(vectors)

array = kmeans_model.labels_
cluster_0 = np.count_nonzero(array==0)
cluster_1 = np.count_nonzero(array==1)
cluster_2 = np.count_nonzero(array==2)
cluster_3 = np.count_nonzero(array==3)
cluster_4 = np.count_nonzero(array==4)
cluster_size_b4_PCA = [cluster_0,cluster_1,cluster_2,cluster_3,cluster_4]

print(cluster_size_b4_PCA)
np.argmax(cluster_size_b4_PCA)


# In[6]:


#calculate SSE for each k
sse = {}
for k in range(0,5):
    sse[k] = kmeans_model.inertia_
    
round(sse[0],2 )   


# In[13]:


pca = PCA(n_components=100, random_state=1,svd_solver='full')
pca.fit(vectors)
x_pca = pca.transform(vectors)
x_pca.shape


# In[14]:


kmeans1 = KMeans(n_clusters=5, random_state=1)
kmeans1.fit(x_pca)

array1 = kmeans1.labels_
cluster1_0 = np.count_nonzero(array1==0)
cluster1_1 = np.count_nonzero(array1==1)
cluster1_2 = np.count_nonzero(array1==2)
cluster1_3 = np.count_nonzero(array1==3)
cluster1_4 = np.count_nonzero(array1==4)
cluster_size_ftr_PCA = [cluster1_0,cluster1_1,cluster1_2,cluster1_3,cluster1_4]
print(cluster_size_ftr_PCA)
np.argmax(cluster_size_ftr_PCA)


# In[16]:



sse1 = kmeans1.inertia_
round(sse1, 2)
    


# In[26]:


det = df['Vector']
vectors = pd.DataFrame(list(det))
kmeans_model = KMeans(n_clusters = 5,random_state=1).fit(vectors)

array = kmeans_model.labels_
cluster_0 = np.count_nonzero(array==0)
cluster_1 = np.count_nonzero(array==1)
cluster_2 = np.count_nonzero(array==2)
cluster_3 = np.count_nonzero(array==3)
cluster_4 = np.count_nonzero(array==4)
cluster_size_b4_PCA = [cluster_0,cluster_1,cluster_2,cluster_3,cluster_4]

print(cluster_size_b4_PCA)
np.argmax(cluster_size_b4_PCA)
documents = df['Preprocessed-Article']
vect = TfidfVectorizer(stop_words='english')
X1 = vect.fit_transform(documents)
model = KMeans(n_clusters=5,random_state=1)
model.fit(X1)
lst2 = []
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vect.get_feature_names()
for i in range(5):
    #print('Cluster %d:' %i)
    for ind in order_centroids[i, :50]:
        #print('%s'%terms[ind])
        lst2.append(' %s' % terms[ind])
print(lst2)
#calculate SSE for each k
sse = {}
for k in range(0,5):
    sse[k] =model.inertia_
round(sse[0],2 )   
#Applying PCA:
pca = PCA(n_components=1, random_state=1)
pca.fit(vectors)
x_pca = pca.transform(vectors)
x_pca
kmeans1 = KMeans(n_clusters=5, random_state=1)
kmeans1.fit(x_pca)
array1 = kmeans1.labels_
cluster1_0 = np.count_nonzero(array1==0)
cluster1_1 = np.count_nonzero(array1==1)
cluster1_2 = np.count_nonzero(array1==2)
cluster1_3 = np.count_nonzero(array1==3)
cluster1_4 = np.count_nonzero(array1==4)
cluster_size_ftr_PCA = [cluster1_0,cluster1_1,cluster1_2,cluster1_3,cluster1_4]
print(cluster_size_ftr_PCA)
np.argmax(cluster_size_ftr_PCA)
sse1 = {}
for k in range(0,5):
    sse1[k] = kmeans1.inertia_
round(sse1[0], 2)
result =[round(sse[0], 2), round(sse1[0], 2), (np.argmax(cluster_size_b4_PCA)+1), cluster_size_b4_PCA[np.argmax(cluster_size_b4_PCA)],(np.argmax(cluster_size_ftr_PCA)+1), cluster_size_ftr_PCA[np.argmax(cluster_size_ftr_PCA), lst2[49]]


# In[14]:


preprocessed = data['Preprocessed-Article']
model_ftr_PCA = KMeans(n_clusters=5,random_state=1)
model_ftr_PCA.fit(vectors)
data['LabelwithoutPCA']=model.labels_
#print(data['LabelwithoutPCA'])
data['LabelafterPCA']=model_ftr_PCA.labels_
#print(data['LabelafterPCA'])
for each,subset in data.groupby('LabelafterPCA'):
    lst3=[]
    lst3.append(''.join(subset['Preprocessed-Article']).lower().encode('utf-8').split()[:50])
lst3
a6 = lst3[0]
print(a6)


# In[ ]:




