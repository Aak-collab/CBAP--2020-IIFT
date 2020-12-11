# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:33:15 2020

@author: Aakash Predator
"""

#COVARIANCE DATA
import numpy as np
import seaborn as sns
import pandas as pd

df = pd.DataFrame(np.random.randint(low=0, high=20, size=(5,2)), columns=["A", "B"])
sns.scatterplot(df.A, df.B)


#Skewness and Kurtosis
from scipy.stats import kurtosis
from scipy.stats import skew
import matplotlib.pyplot as plt
import scipy.stats as stats
plt.style.use('ggplot')

data = np.random.normal(0,1,1000000)
data

np.var(data)
np.mean(data)
np.std(data)

plt.hist(data, bins=60)

print('mean:' , np.mean(data))
print('var: ', np.var(data))
print('skew: ', skew(data))
print('kurtosis: ' , kurtosis(data))

## ONE SAMPLE TEST
pop_marks = np.random.normal(loc=55, scale=12, size=10000)
pop_marks

sample = pop_marks[400:429]
sample

dfs = pd.Series(sample)
dfs

plt.hist(pop_marks)
plt.hist(sample)

dfs.skew()
dfs.kurtosis()
#Negatively skewd to the left
np.mean(sample)
np.std(sample)

assumed_mean = np.mean(pop_marks)

ttestS1 = stats.ttest_1samp(a=sample, popmean=assumed_mean)
ttestS1
