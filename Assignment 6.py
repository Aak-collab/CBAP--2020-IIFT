#!/usr/bin/env python
# coding: utf-8

# # Assignment 6 : code by me

# In[ ]:


import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

train=pd.read_csv('/data/training/Pacific_train.csv')
test=pd.read_csv('/data/test/Pacific_test.csv')
#feature engineer latitude , longitude:
def conv(deg):
    if deg[-1:]==('W'or'N'):
        return float(deg[:-1])
    else:
        return -1*float(deg[:-1])
lat = train.Latitude
lat_array = np.array(lat)
lat_array
for i in range(len(lat_array)):
    lat_array[i]=conv(lat_array[i])
train.Latitude = pd.DataFrame(lat_array)
long_ = np.array(train.Longitude)
for i in range(len(long_)):
    long_[i]=conv(long_[i])
train.Longitude = pd.DataFrame(long_)
train.Latitude = train.Latitude.astype(float)
train.Longitude = train.Longitude.astype(float)
print(train.info())
lat_test = np.array(test.Latitude)
for i in range(len(lat_test)):
    lat_test[i]=conv(lat_test[i])
test.Latitude = pd.DataFrame(lat_test)
long_test = np.array(test.Longitude)
for i in range(len(long_test)):
    long_test[i]=conv(long_test[i])
test.Longitude = pd.DataFrame(long_test)
test.Latitude = test.Latitude.astype(float)
test.Longitude = test.Longitude.astype(float)
print(test.info())
le=LabelEncoder()
le.fit(train.Event)
train.Event=le.transform(train.Event)
le.fit(test.Event)
test.Event=le.transform(test.Event)
train['Category_id'] = train.Status.factorize()[0]
test['Category_id'] = test.Status.factorize()[0]
train=train.drop(['ID','Name','Date','Status'], axis=1)
test=test.drop(['ID','Name','Date','Status'], axis=1)
X_train=train.drop(['Category_id'], axis=1)
X_test=test.drop(['Category_id'], axis=1)
Y_train=train.Category_id
Y_test=test.Category_id
#fitting the Decision Tree Model:
tree_clf=DecisionTreeClassifier(max_depth=7, criterion='entropy')
selector=RFE(tree_clf, step=1)
selector.fit(X_train,Y_train)
rfe_support=selector.get_support()
rfe_feature = X_train.loc[:,rfe_support].columns.tolist()
print(rfe_feature)
X_train = X_train.drop(['Event','Low Wind SE', 'Low Wind SW', 'Low Wind SW','Moderate Wind NE','Moderate Wind SE','Moderate Wind SW','Moderate Wind NW','High Wind NE','High Wind NW'], axis=1)
#Y_train = Y_train.drop(['Time', 'Event', 'Longitude', 'Minimum Pressure', 'Low Wind NE',
                                #'Low Wind SE', 'Low Wind SW', 'Low Wind SW', 'Low Wind NW'], axis=1 )
X_test = X_test.drop(['Event','Low Wind SE', 'Low Wind SW', 'Low Wind SW','Moderate Wind NE','Moderate Wind SE','Moderate Wind SW','Moderate Wind NW','High Wind NE','High Wind NW'], axis=1)                                 
#grd = GridSearchCV(DecisionTreeClassifier(),{'criterion':['gini','entropy'],'max_depth':[1,2,3,4,5,6,7,8,9]},cv=10)
#grd.fit(X_train,Y_train)
#print(grd.best_params_)
tree_clf.fit(X_train, Y_train)
y_pred_dt=tree_clf.predict(X_test)
dec_tree_scores = [recall_score(Y_test, y_pred_dt, average='weighted'), precision_score(Y_test, y_pred_dt, average='micro'), 
                        accuracy_score(Y_test, y_pred_dt)]
accuracy_score_dt= accuracy_score(Y_test,y_pred_dt)
print('Decision tree model accuracy:',accuracy_score_dt)
#Fitting the Random Forest Model:
#grd_rf = GridSearchCV(RandomForestClassifier(), {'n_estimators':[100,200], 
     # 'criterion':['gini','entropy']},cv=10)
#grd_rf.fit(X_train,Y_train)
#print(grd_rf.best_params_)
#rf_clf = RandomForestClassifier(max_depth=2, n_estimators=100)
#rf_clf.fit(X_train,Y_train)
#y_pred_rf = rf_clf.predict(X_test)
#accuracy_score_rf = accuracy_score(Y_test,y_pred_rf)
#print("Random Forests Model accuracy:",accuracy_score_rf)
#Fitting GaussianNB:
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
y_pred_gnb = gnb.predict(X_test)
accuracy_score_gnb = accuracy_score(Y_test, y_pred_gnb)
print("Naive Bayes model accuracy:",accuracy_score_gnb)


#result=['GaussianNB', 0.7]
#result=pd.DataFrame(result)
#writing output to output.csv
#result.to_csv('/code/output/output.csv', header=False, index=False)


# In[8]:


import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

df=pd.read_csv('C:/Users/Aakash Predator/OneDrive/Desktop/Datasets/Pacific_train.csv')


arr1 = np.array(df.Latitude)
for i in range(len(arr1)):
    arr1[i]=arr1[i].replace('N','')
    
arr2 = np.array(df.Longitude)
for i in range(len(arr2)):
    arr2[i]=arr2[i].replace('W','')
df.Latitude = pd.DataFrame(arr1)
df.Longitude = pd.DataFrame(arr2) 
    
arr3 = np.array(df.Longitude)
for i in range(len(arr3)):
    arr3[i]=arr3[i].replace('E','')

df.Longitude = pd.DataFrame(arr3)
df.Latitude = df.Latitude.astype(float)
df.Longitude = df.Longitude.astype(float)

df['Category_id'] =df.Status.factorize()[0]
df = df[['Time', 'Latitude', 'Longitude', 'Maximum Wind', 'Minimum Pressure','Category_id']]
X = df.drop(['Category_id'],axis=1)
Y = df.Category_id



# In[22]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=3)
#fitting the Decision Tree Model:
tree_clf=DecisionTreeClassifier(max_depth=100, criterion='entropy')
              
#grd = GridSearchCV(DecisionTreeClassifier(),{'criterion':['gini','entropy'],'max_depth':[1,2,3,4,5,6,7,8,9]},cv=10)
#grd.fit(X_train,Y_train)
#print(grd.best_params_)
tree_clf.fit(X_train, Y_train)
y_pred_dt=tree_clf.predict(X_test)
dec_tree_scores = [recall_score(Y_test, y_pred_dt, average='weighted'), precision_score(Y_test, y_pred_dt, average='micro'), 
                        accuracy_score(Y_test, y_pred_dt)]
accuracy_score_dt= accuracy_score(Y_test,y_pred_dt)
print('Decision tree model accuracy:',accuracy_score_dt)
#Fitting the Random Forest Model:
rf_clf = RandomForestClassifier(max_depth=200, n_estimators=100)
rf_clf.fit(X_train,Y_train)
y_pred_rf = rf_clf.predict(X_test)
accuracy_score_rf = accuracy_score(Y_test,y_pred_rf)
print("Random Forests Model accuracy:",accuracy_score_rf)
#Fitting GaussianNB:
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
y_pred_gnb = gnb.predict(X_test)
accuracy_score_gnb = accuracy_score(Y_test, y_pred_gnb)
print("Naive Bayes model accuracy:",accuracy_score_gnb)


#result=['GaussianNB', 0.7]
#result=pd.DataFrame(result)
#writing output to output.csv
#result.to_csv('/code/output/output.csv', header=False, index=False)


# In[ ]:




