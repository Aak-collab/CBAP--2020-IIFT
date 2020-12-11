#!/usr/bin/env python
# coding: utf-8

# **MID Program Project**

# In[2]:


#loading required libraries:
import numpy as np
import pandas as pd
import calendar
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from scipy import stats
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor


# In[31]:


#reading the datasets and merging them
df_train = pd.read_csv('C:/Users/Aakash Predator/Downloads/mid prog project1/Dataset/train.csv')
df_train
df_train_label = pd.read_csv('C:/Users/Aakash Predator/Downloads/mid prog project1/Dataset/train_label.csv',header=None)
df_train['Booking']=df_train_label[0]
df_train=df_train.dropna()
df_train
df_test = pd.read_csv('C:/Users/Aakash Predator/Downloads/mid prog project1/Dataset/test.csv')
df_test_label = pd.read_csv('C:/Users/Aakash Predator/Downloads/mid prog project1/Dataset/test_label.csv', header=None)
df_test['Booking']=df_test_label[0]
df_test=df_test.dropna()
df_test


# In[32]:


#plt.scatter(df_train['Booking'],df_train['windspeed'])
#df_train.weather.unique()
#feature engineering
df_train['date']=df_train['datetime'].apply(lambda x: x.split()[0])
df_train['hour']=df_train['datetime'].apply(lambda x: x.split()[1].split(':')[0])
df_test['date']=df_test['datetime'].apply(lambda x: x.split()[0])
df_test['hour']=df_test['datetime'].apply(lambda x: x.split()[1].split(':')[0])
df_train['date'] = pd.to_datetime(df_train.date, infer_datetime_format=True)
df_test['date'] = pd.to_datetime(df_test.date, infer_datetime_format=True)
df_train['month']=df_train.date.dt.month
df_train.weather=df_train.weather.apply(lambda x:x.split('+'))
df_train['weather_1']=df_train.weather.apply(lambda x:x[0])
df_train['weather_2']=df_train.weather.apply(lambda x:x[-1])
df_test['month']=df_test.date.dt.month
df_test.weather=df_test.weather.apply(lambda x:x.split('+'))
df_test['weather_1']=df_test.weather.apply(lambda x:x[0])
df_test['weather_2']=df_test.weather.apply(lambda x:x[-1])
df_train=df_train.drop(columns='weather')
df_test=df_test.drop(columns='weather')
df_train= df_train.drop(columns='temp')
df_test = df_test.drop(columns='temp')
month_arr=np.array(df_train.date.dt.month)
day_arr=np.array(df_train.date.dt.day)
year_arr=np.array(df_train.date.dt.year)
weekday=[]
for i in range(len(month_arr)):
    weekday.append(calendar.weekday(year_arr[i],month_arr[i],day_arr[i]))
weekday=pd.DataFrame(weekday)
df_train['weekday']=weekday
df_train.head()
month_arr1=np.array(df_test.date.dt.month)
day_arr1=np.array(df_test.date.dt.day)
year_arr1=np.array(df_test.date.dt.year)
weekday1=[]
for i in range(len(month_arr1)):
    weekday1.append(calendar.weekday(year_arr1[i],month_arr1[i],day_arr1[i]))
weekday1=pd.DataFrame(weekday1)
df_test['weekday']=weekday1
df_test.head()
df_train=df_train.drop(columns=['datetime','date'],axis=1)  
df_test=df_test.drop(columns=['datetime','date'],axis=1)
#sns.boxplot(df_train.weekday,df_train.Booking):
df_train.drop(df_train[df_train.holiday==df_train.workingday].index, inplace=True)
#label encoding of categorical features:
le=LabelEncoder()
df_train.season =le.fit_transform(df_train.season)
df_train.weather_1 = le.fit_transform(df_train.weather_1)
df_train.weather_2=le.fit_transform(df_train.weather_2)
df_test.season =le.fit_transform(df_test.season)
df_test.weather_1 = le.fit_transform(df_test.weather_1)
df_test.weather_2=le.fit_transform(df_test.weather_2)


# In[33]:


#removing outliers
df_train_new=df_train[(df_train.Booking<=286)]
df_train=df_train_new
df_train.hour = df_train.hour.astype('float')
df_test.hour = df_test.hour.astype('float')


# In[34]:


#visualizing data
sns.boxplot(df_train.weather_2,df_train.Booking)


# In[35]:


sns.boxplot(df_train.weather_1,df_train.Booking)


# In[36]:


sns.boxplot(df_train.month,df_train.Booking)


# In[37]:


sns.boxplot(df_train.hour,df_train.Booking)


# In[38]:


sns.boxplot(df_train.windspeed,df_train.Booking)


# In[39]:


sns.boxplot(df_train.season,df_train.Booking)


# In[40]:


sns.boxplot(df_train.workingday,df_train.Booking)


# In[41]:


sns.boxplot(df_train.holiday,df_train.Booking)


# In[42]:


sns.boxplot(df_train.humidity,df_train.Booking)


# In[60]:


#dividing test and train data:
Y_train = df_train.Booking
X_train = df_train.drop(columns=['Booking','season','holiday','workingday','weather_1','weather_2','weekday'])
Y_test = df_test.Booking
X_test = df_test.drop(columns=['Booking','season','holiday','workingday','weather_1','weather_2','weekday'])
X_train


# In[44]:


#hhyperparameter tuning for random forest regressor:
param_grid = {'n_estimators':[10,100,10], 'criterion' :['mae','mse'], 'max_depth':[1,5,1]}
rfe = RandomForestRegressor()
rfe_tuned = GridSearchCV(rfe, param_grid=param_grid,cv=5)


# In[45]:


rfe_model = rfe_tuned.fit(X_train,Y_train)


# In[46]:


y_pred = rfe_model.predict(X_test)
mse_rfe = mean_squared_error(Y_test, y_pred)


# In[47]:


print(np.sqrt(mse_rfe))


# In[48]:


r2_score_rfe = r2_score(Y_test,y_pred)
r2_score_rfe


# In[49]:


#model for decisionTreeRegressor:
parameters  = {'criterion':['mse','friedman_mse','mae'], 'max_depth':[1,10,1], 'max_features':['auto',
               'sqrt','log2'],'random_state':[1] }
dtr = DecisionTreeRegressor()
dtr_tuned =GridSearchCV(dtr,parameters,cv=5)


# In[50]:


dtr_model = dtr_tuned.fit(X_train,Y_train)


# In[51]:


Y_pred_dtr = dtr_model.predict(X_test)
mse_dtr = mean_squared_error(Y_test,Y_pred_dtr)
r2_score_dtr=r2_score(Y_test,Y_pred_dtr)
mse_dtr
r2_score_dtr


# In[52]:


xg = XGBRegressor()
param_grid_xgb = {'eta':[0.1,0.2,0.3], 'min_child_weight':[0,1,2,3,4,5], 'max_depth':[3,4,5,6,7,8,9],
                 'alpha':[0,1], 'scale_pos_weight':[0,1,2,3]}
xg_tuned = GridSearchCV(xg,param_grid_xgb,cv=5)
xg_model=xg_tuned.fit(X_train,Y_train)


# In[53]:


y_pred_boosted = xg_model.predict(X_test)
y_pred_boosted


# In[54]:


mse_xgboost = mean_squared_error(Y_test,y_pred_boosted)
r2_score_xgboost = r2_score(Y_test,y_pred_boosted)


# In[55]:


np.sqrt(mse_xgboost)


# In[56]:


r2_score_xgboost


# In[57]:


dtr_tuned.best_params_


# In[58]:


rfe_tuned.best_params_


# In[59]:


xg_tuned.best_params_


# In[ ]:




