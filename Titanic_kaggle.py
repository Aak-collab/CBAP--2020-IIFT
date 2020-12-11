#!/usr/bin/env python
# coding: utf-8

# # Kaggle-Titanic data

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


# **Reading the Data**

# In[16]:


train = pd.read_csv('C:/Users/Aakash Predator/Downloads/TITANIC/train.csv')
train.head(5)
test = pd.read_csv('C:/Users/Aakash Predator/Downloads/TITANIC/test.csv')
test.head(5)


# In[ ]:





# In[15]:


test.isnull().sum()


# In[7]:


def bar(feature):
    survived=train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index=['Survived','Dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))


# In[8]:


bar('Parch')


# In[10]:


bar('SibSp')


# In[18]:


le = LabelEncoder()
train.Sex=le.fit_transform(train['Sex'])
test.Sex=le.fit_transform(test['Sex'])


# In[19]:


train


# In[12]:


bar('Sex')


# This chart infers that women survived more than men.
# 

# In[31]:


train.Embarked=train.Embarked.astype('str')
test.Embarked=test.Embarked.astype('str')


# In[32]:


train.Embarked = le.fit_transform(train.Embarked)
test.Embarked = le.fit_transform(test.Embarked)


# In[14]:


bar('Embarked')


# In[20]:


bar('Pclass')


# In[45]:


train
x_train = train.drop(['Survived'],axis=1)
y_train = train.Survived
x_train
x_train.shape[1]


# In[36]:


#plt.scatter(x_train.Fare,y_train)


# **Trying One Hot Encoding with the data:**

# In[ ]:





# In[ ]:





# In[46]:


test['Fare']=test.Fare.astype('float')


# In[47]:


test.Fare


# In[ ]:


test.Fare.fillna('0',inplace=True)


# In[52]:


test.Fare = test.Fare.dropna().isnull().sum()


# In[ ]:





# # Decision Tree Classifier

# In[53]:


dtc = DecisionTreeClassifier()


# In[54]:


dtc.fit(x_train,y_train)


# In[57]:


y_pred_dtc=dtc.predict(test)
y_pred_dtc = pd.DataFrame(y_pred_dtc,columns=['Survived'])
y_pred_dtc['PassengerId']= y_true
y_pred_dtc.to_csv('submission_dtc.csv',index=False)


# In[58]:


train_pred_dtc = dtc.predict(x_train)
train_pred_dtc
train_accuracy_dtc=accuracy_score(y_train,train_pred_dtc)
train_accuracy_dtc


# In[56]:


y_true = pd.read_csv('C:/Users/Aakash Predator/Downloads/TITANIC/gender_submission.csv')
y_true.drop(['Survived'],axis=1,inplace=True)


# # Random Forest Classifer

# In[59]:


rfc = RandomForestClassifier(criterion='gini')
rfc.fit(x_train,y_train)


# In[60]:


y_pred_rfc = rfc.predict(test)
train_pred_rfc = rfc.predict(x_train)
train_pred_rfc
train_accuracy_rfc=accuracy_score(y_train,train_pred_rfc)
train_accuracy_rfc


# In[61]:


y_pred_rfc = pd.DataFrame(y_pred_rfc,columns=['Survived'])
y_pred_rfc['PassengerId']= y_true
y_pred_rfc.to_csv('submission.csv',index=False)
y_pred_rfc.shape[0]


# # AdaBoost Classifier

# In[62]:


adc = AdaBoostClassifier()
adc.fit(x_train,y_train)
y_pred_adc = adc.predict(test)
y_pred_adc = pd.DataFrame(y_pred_adc,columns=['Survived'])
y_pred_adc['PassengerId']= y_true
y_pred_adc.to_csv('submission_ada.csv',index=False)


# In[63]:


train_pred_adc = adc.predict(x_train)
train_pred_adc
train_accuracy_adc=accuracy_score(y_train,train_pred_adc)
train_accuracy_adc


# # XGboost classifier
# 

# In[64]:


from xgboost import XGBClassifier
xgb = XGBClassifier(base_estimator=rfc)
xgb.fit(x_train,y_train)
y_pred_xgb=xgb.predict(test)
y_pred_xgb


# In[65]:


y_pred_xgb = pd.DataFrame(y_pred_xgb,columns=['Survived'])
y_pred_xgb['PassengerId']= y_true
y_pred_xgb.to_csv('submission_xgb.csv',index=False)


# In[66]:


train_pred_xgb=xgb.predict(x_train)
accuracy_xgb=accuracy_score(y_train,train_pred_xgb)
accuracy_xgb


# In[ ]:





# In[ ]:




