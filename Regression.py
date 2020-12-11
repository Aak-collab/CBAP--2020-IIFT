# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:30:28 2020

@author: Aakash Predator
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = {'Price': list(np.random.randint(100,200,8)), "Area": list(np.random.randint(400,800,8)) }
df = pd.DataFrame(df)
df

plt.scatter(df.Price, df.Area)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(df.Area.values, df.Price.values) #accepts numpy
x = df.Area.values
y = df.Price.values
x.shape
x = x.reshape(-1,1)
x.shape
model.fit(x,y)
y_pred = model.predict(x)
y_pred

plt.scatter(x,y)
plt.scatter(x,y_pred)

x_test = np.array([2500,2700, 2950, 3300]).reshape(-1,1)
x_test.shape

y_pred1 = model.predict(x_test)
y_pred1

plt.scatter(x,y)
plt.scatter(x,y_pred)
plt.scatter(x_test, y_pred1)

#checking metrics

r_sq = model.score(x,y)
r_sq

#r squared values shows the capability of the model
from pydataset import data
mtcars = data('mtcars')
mtcars

mtcars.columns
x=mtcars.hp.values.reshape(-1,1)
y=mtcars.disp.values.reshape(-1,1)
x.shape
y.shape

model2 = LinearRegression()
model2.fit(x,y)

y_pred = model2.predict(x)

plt.scatter(x,y)
plt.scatter(x,y_pred)

r_sq = model2.score(x,y)
r_sq

#multi Linear Regression
x= mtcars[['mpg', 'hp']].values
x.shape

y= mtcars['disp'].values
y.shape

model.fit(x,y)

y_pred = model.predict(x)

r_sq = model.score(x,y)
r_sq
print('R squared is :', round((r_sq*100),2), "%")

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

x_train.shape
x_test.shape
y_train.shape
y_test.shape

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_pred
r_sq = model.score(x_test, y_test)
r_sq

# Example 4 ( library for statisitical purposes)
import statsmodels.api as sm
from statsmodels.formula.api import ols
MTmodel1 = ols("mpg~ wt + hp", data=mtcars).fit()
print(MTmodel1.summary())

pred_m1 = MTmodel1.predict()

fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_ccpr_grid(MTmodel1, fig=fig)

#r2 score, mean squared error

from sklearn.metrics import r2_score, mean_squared_error

r2_score(y_test, y_pred)
mean_squared_error(y_test, y_pred)

#r2 is a good enough parameter

