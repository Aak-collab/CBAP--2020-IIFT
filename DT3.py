# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 20:41:09 2020

@author: vikas
"""

# -*- coding: utf-8 -*-

import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


df = pd.read_csv("loan_data.csv")

df.columns

df = df.drop(['credit.policy', 'purpose'], axis=1)


y = df['not.fully.paid']

y


df = df.drop(['not.fully.paid'], axis =1)

X = df

df.dtypes

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)
y_pred
y_test

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))




y_pred_proba = logreg.predict_proba(X_test)[::,1]
y_pred_proba
y_pred_proba.shape


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

#together

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show();





from sklearn.tree import DecisionTreeClassifier

clsModel = DecisionTreeClassifier()  #model with parameter

clsModel.fit(X_train, y_train)

#predict
ypred = clsModel.predict(X_test)
ypred



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))




y_pred_proba = clsModel.predict_proba(X_test)[::,1]
y_pred_proba
y_pred_proba.shape


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

#together

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show();







#Example




#Example


# -*- coding: utf-8 -*-

import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
from sklearn.metrics import roc_curve, auc


df = pd.read_csv("Social_Network_Ads.csv")

df.columns

df = df.drop(['User ID', 'Gender', 'Age'], axis=1)


y = df['Purchased']

y


df = df.drop(['Purchased'], axis =1)

X = df

df.dtypes

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.tree import DecisionTreeClassifier

clsModel = DecisionTreeClassifier()  #model with parameter

clsModel.fit(X_train, y_train)

#predict
y_pred = clsModel.predict(X_test)
ypred.shape
y_test.shape

y_test, y_pred

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))




y_pred_proba = clsModel.predict_proba(X_test)[::,1]
y_pred_proba
y_pred_proba.shape


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

#together

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show();


