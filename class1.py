# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 20:54:51 2020

@author: Aakash Predator
"""

s1 = input(" my name is")
print(s1, "rocks")
s2 = input( "enter your age")
print("my name is {0} and my age is {1}".format(s1,s2))
import pandas as pd
a = 10.5
f_to_i = int(a)
print(f_to_i)
b = "1"
c = int(b)
print(b)

r1 = range(10)
lr1 = list(r1)
print(lr1)
r2 = range(5,10)
lr2 = list(r2)
print(lr2)
r2
r3 = range(0,100,2)
print(r3)
lr3 = list(r3)
print(lr3)
l4 = [1,2,1,1,3,'vikas',True,'vikas','vikas']
print(l4.count('vikas'))
print(l4.count(1))
l5 = [1,2,3,4]
#l5[4]= 20
l5.append(7)
l5.append('vikas')
print(l5)
l5.remove(1)
l5
#sort function
fruits = ['apple',"cherry", 'banana']
fruits.sort()
fruits
fruits.insert(1,'mango')
fruits
fruits.sort()
fruits
l = [1,2,3,"a",'b','z','xc']
l.sort()
l
#not supported sort function for 
#different data types in the list

#adding or concatenating two lists
l = fruits + l
l

l10 = list(range(1,20))
l10
#reversing list
fruits[::-1]
print(fruits[::-1])

#Sets ( similar to set theory)
#Unindexed
s1 = {1,2,3}
type(s1)
s2 = {1,2.5, True, "khullar", "vikas" }
print(s2)
s2[1]



# TUples
#immutable, collection,, ordered, round bracket
t1 = ()
type(t1)
t1 = (1,5,3,2,6)
print(t1)
t1[1]
# has no attribute append or remove

#Enumerate: 

L1 = ['vikas', 'aman', 'raman']
e1 = enumerate(L1)
e1
for i in L1:
    print(i)

for i in e1:
    print(i)

e1 = enumerate(L1, start=100)

for i in e1:
    print(i)
    
l2 = list(range(1,100))
e2 = enumerate(l2)
for i in e2:
    print(i)
    
cnt = 0
for i in e2:
    print(i)
    if(cnt==50):
        break
    cnt+=1

#Frozen set
fz1 = frozenset([1,2,3,4,5,6,7])
type(fz1)
fz1
fz1.add(10) #cant be added
s1 = {1,2,3,4}
fz1.union(s1)
fz1.intersection(s1)
fz1.difference(s1)

#Zip data type
name = ['vikas', 'Vikrant', 'Abhay', 'amit']
rno = [111,112,113,114]
marks = [80,50,99,95]
z1 = zip(name,rno,marks)
z1
l1 = list(z1)
l1 #returnng a list of tuples

for i,j,k in z1:
    print(i,j,k)
    
#Unzipping
namz, rnoz, markz = zip(*z1)
namz, rnoz, markz 


#Looping:
L1 = ['A', "B", "C","D"]
for i in range(len(L1)):
    print("Index",i,"has element",L1[i])

for i in range(0,11):
    print("2*",i,"=",2*i)


#Loop inside a loop:
for j in range(1,11):
    for i in range(0,11):
        print(j,"*",i,"=",j*i)
 
#Loops with conditions:
teamA = ["India", "Pakistan", "Austrailia"]
for i in teamA:
    if i=="Pakistan":
        print(i, "Inner")
      

i= 1
while(i<10):
    
    marks = float(input("enter Marks"))
    if marks>85 :
        print("A")
    elif (marks<85 and marks>=70):
        print("B")
    elif(marks<70 and marks>=50):
        print("C")
    i+=1
    
    
#using functions

a=10
b=20

def oper1():
    print(a+b)
    
oper1()
def oper(a,b):
    print(a+b)
    
    
oper(2,3)

def details(name, email, batch="not available"):
    print(name)
    print(email)
    print(batch)
    
details("Aakash", "aakash.basu@redfmail.com")    

def totalsale(sale=0, refund=0):
    return( sale-refund)
    
    
ret=totalsale(100,80)
ret

def reverse(s):
    return(s[::-1])
    
reverse("vikas")

def max_in_list(lst):
    max_1 = 0
    for n in lst:
        if n>max_1:
            max_1=n
    return max_1

lst1 = [1,2,3,4,5,6,7,8,9]

m = max_in_list(lst1)
m

"""

STATISTICAL ANALYSIS


"""
#Using RANDOM LIBRARY 
import random
L1 = [1,2,3,4,5,60]
print(random.choice(L1))
x= random.randint(0,10)
x

import random as rd
rd.randint(10,50)

rd.choice(L1)

professions = ["scientist", "philosopher", "engineer" ] 
rd.choice(professions)
# choosing k number randomly from the list
samples = rd.choices(professions, k=10)
samples


weight_set = {20,35,45,65,82}
weight = rd.choice(tuple(weight_set))
weight
weights = rd.choices(tuple(weight_set),k=5 )
weights
weight = rd.choices(list(weight_set), k =5)
weight

#Dictionary data needs to be converted to tuple or list for this function
for i in range(5):
    rd.seed(1) #generating same random number 5 times
    print(rd.randint(1,1000))
    
    
for i in range(5):
    print(rd.randint(1,1000))
    
# NUMPY LIBRARY OPERATIONS for matrix ops
# LIbrary can manage data in the form of rows and columns

import numpy as np
np.random.randint(100,1000)
x1 = np.random.randint(100,200,size=10)
x1
#difference between list and array
#array is the type of data controlled ny numpy
#List can contain heterogenous data  Array has homogenous data

x1.shape
#creating an array matrix
x2 = np.random.randint(100,200,size=(3,4))
x2
x2.shape
#array can be accessed by index
x1[0]
x2[1]
x1[1:5]
x2[0][0]
x2[:][:]

#creatinga  3d array
x3 = np.random.randint(100,200, size=(2,3,4))
x3

#Slicing from a matrix

x4 =np.random.randint(10,20,size=(5,5))
x4
x4[:2,3:]
x4[:2,-2:]

#selecting alternative rows and columns
x1[::2] #skipping by 1

x4[::2]
x4[:,::5]

ar1 = np.arange(20)
ar1

#pulling the diagonal
np.diagonal(x4)

for i in range(x4.shape[0]):
    for j in range(x4.shape[1]):
        if (i==j):
            print(x4[i][j])
            
            
x4.diagonal()

#Reshaping array
x5 = np.arange(0,20)
x5
x5.reshape(4,5)

#creating an array of zeroes
a = np.zeros((2,4))
a
#by default float array is being created
#creating an array of ones
b = np.ones((4,2))
b
#creating identity matrix
c=np.eye(3,3)
c

#dividing a line into equal portions
ls1 = np.linspace(0,10,num=3)
ls1

#creating an array with tuples
x6 = np.array([(1,2,3),(4,5,6)])
x6
#creating an array with integer elements
x8 = np.array([1,2,4], dtype="int")
x8


#some statistical operations
x9 = np.random.randint(40,100, size=100)
x9
m = np.max(x9)
m
min(x9)
np.mean(x9)
np.median(x9)
np.std(x9)
#loop for average:
s=0
for i in x9:
    s=s+i
    
avg=s/len(x9)
avg


nl1 = np.linspace(0,10,5)
np.diff(nl1)  #difference between the elements
np.round(nl1) # rounds off the elements
np.floor(nl1) #floor function
np.ceil(nl1)  #ceiling function
np.round([1.2,1.6]) #rounded off (nearest integer)
np.trunc([1.2,1.6]) # removing the deicimal part
np.floor([1.2,1.6]) #round off to lowest integer
np.ceil([1.2,1.6])  # round off to highest integer


#CONCATENATE FUNCTION
x4= np.array([1,2,3,4,5,6])
x5=np.ones(10)
np.concatenate([x4,x5])
#row wise concatenation
x4=x4.reshape(3,2)
x5=x5.reshape(5,2)
np.concatenate([x4,x5], axis=0)
#column wise concatenation
x4=x4.reshape(2,3)
x5=x5.reshape(2,5)
x6= np.concatenate([x4,x5], axis=1)
x6 = x6.astype("int")
x6

#vertical stacking
x4=x4.reshape(3,2)
x5=x5.reshape(5,2)
np.vstack([x4,x5])

#horizontal stacking
np.hstack([x4,x5])

#splitting
x7 = np.arange(10,20)
x7
#to be done ...

x8 =np.random.randint(5,100, size=(5,5))
x8
np.mean(x8)#Overall mean
np.mean(x8, axis=0) #one outcome for each row
np.mean(x8, axis=1) #one outcome for each column
np.equal(x8,34) #checking which number in the array is eaula to the given number
 
np.sum(np.equal(x8,52)) #checking how many numbers are equal to the given number
np.sum(np.greater(x8,52))#how manny values are greater than the given value
np.sum(np.less(x8,52)) #how many values are less than 52

x8 < 52
np.all(x8>4) #are all values greater than 4?
np.any(x8>70) #is there any value s greater than 70

x9 = np.sort(x8) #sortef row wise
x9
x10 = np.sort(x8, axis=1) #sorted coulmn wise
x10

x = np.arange(1,13).reshape(3,4)
x
x.T

#PLOTTIGN A GRAPH WITH MATPLOTLIB

from matplotlib import pyplot as plt
#normally distributed random data
n1 = np.random.normal(100,20,1000)
"""
what is the mean, standard deviation??
np.random.normal(mean, stddev, size)
"""
#plotting a histogram to show the normal distribution
l1 = list(range(50,150))
l2  = np.linspace(min(n1), max(n1), 10)
plt.hist(n1, bins=l1)
plt.title("histogram")
plt.show()

# PANDAS LIBRARY

import pandas as pd
help(pd)
pip install pydataset
from pydataset import data

#Checking the available datasets

data("")
#assignng a dataset in a variable
mt = data('mtcars')
mt
Ap = data("AirPassengers")
Ap
type(mt)


mt.head()
mt.tail()

#saving file to location
mt.to_csv("mt.csv")
#calling the file
#csv = Comma separated values ( lightest form of dataset)

mtcars = pd.read_csv("mt.csv")
mtcars

s = pd.Series(range(1,10))
s
s1 = pd.Series(range(101,133))
mt1 = mt.set_index(s1)
mt1
#finding the shape
mt1.shape

#creating a synthetic dataset
st = "Student_"
name = []
for i in range(65,91):
    name.append(st+chr(i))
    
name
print(name)

#Assgning own indexes
ps2 =pd.Series([1,34,23,32,55], index=['a','b','c','d','e'])
ps2
"""
Indexing is of 2 types:
Front end indexing
Back end indexing by Pandas (cant be changed)

"""
#fetching indexes

ps2['a']

ps2['c':'d']

#loc concept

ps2 = pd.Series([1,34,23,32,55], index=[5,6,7,8,9])

ps2.loc[5:6] #getting values  by the index values user defined

ps2.iloc[0:3]#getting system defined index values

#filtering dataframes

ps3 = pd.Series(range(100,10000))
ps3[ps3 > 999]
ps3[(ps3 < 990) & (ps3>345)]

course = pd.Series(['Btech', "Mtech", ' MBA', 'PhD'])
strength = pd.Series([100,50,120,20])
fees = pd.Series([2.5,4,4.5,3])
 pd1 = pd.DataFrame([course, strength, fees])
pd1

#best way ot create a dictionary" use a dictionary
pd2 = pd.DataFrame({'course':course, "strength": strength, "fees":fees})
pd2
type(pd2)
pd2['course']
pd2.course
pd2.index
pd2.columns
#assigning indexes
pd2.index
pd2''course'
pd2 = pd.DataFrame({[course.strength, fees]}, columns=["A","B","C"."D"])
pd2.head()
pd2[pd2["course"]=="Mtech"]
pd2.drop(pd2['course'=='Mtech']).index
pd3 = pd.DataFrame({'course':course, "strength": strength, "fees":fees})
id = pd3[pd3['course']=='Mtech'].index
pd3.drop(id)

#Statistics
pd3.fees.sum()
pd3.fees.max()
pd3.fees.min()
pd2.fees.std()

#Dropping null values in a dataset
pd4 = pd.DataFrame([['dhiraj', 50, 'M', 10000, None,1], 
                    [None, None, None, None, None,2], 
                    ['kanika', 28, None, 5000, None,3],
                    ['tanvi', 20, 'F', None, None,4], 
                    [None, None, None, None, None,5],
                    [1, 2, 3, 4, 5,6]])
pd4

pd4.dropna(axis=0, how='any')
pd4.dropna(axis =1 , how='all')
pd4.dropna(axis=1, how='any')

import pandas as pd
import numpy as np
pd.set_option('display.max_columns',15)#to show as many columns as required

#DENCO CASE STUDY
df=pd.read_csv('20denco.csv')
df.head()
df.describe()
#numpy counts
df.region.value_counts().plot(kind='bar')
df.custname.value_counts().sort_values(ascending=False).head(5)
#pandas way
df.groupby('custname').size()
df.groupby('custname').size().sort_values(ascending=False).head(5)
#Customer wise revenue
df.groupby('custname').aggregate({'revenue':np.sum})

#Discussing Matplotlib
#librar name
import matplotlib.pyplot as   plt

plt.plot(Year,Unemployment_Rate)
plt.title("year wise unemployment")
plt.xlabel("Year")
plt.ylabel('Unemployment Rate')

Year = [1910, 1930, 1940, 1950 , 1960, 1970, 1980, 1990, 2000, 2010]
Unemployment_Rate = [9.8, 12 , 8, 7.2, 6.9, 7, 6.5, 6.2, 5.5,6.3]

#Coloring
plt.plot()
plt.plot(Year, Unemployment_Rate, color='k')
plt.title("year wise unemployment")
plt.xlabel("Year")
plt.ylabel('Unemployment Rate')

plt.plot()
plt.plot(Year, Unemployment_Rate, color='#FFFF00', marker = 'o')
plt.title("year wise unemployment")
plt.xlabel("Year")
plt.ylabel('Unemployment Rate')

#go to online color palette codes to find color hex code

Data = {"Year": Year, "Unemployment_Rate": Unemployment_Rate}
df = pd.DataFrame(Data)
df

country = ['USA', 'Canada', 'Germany', 'UK', "France"]
GDP_per_capita = [45000, 42000, 52000, 49000, 47000]
New_colors = ['green', 'blue', 'purple', 'brown']
plt.bar(country, GDP_per_capita,color=New_colors)
plt.title('Country vs GDP per capita')
plt.xlabel('Country', fontsize=14)
plt.ylabel('GDP per Capita', fontsize=14)
plt.grid(True)


#Subplots
labels = ['G1', "G2", 'G3', 'G4', 'G5']
men_means = [10,35,30,35,27]
women_means= [25,32,34,20,25]
men_std = [2,3,4,1,2]
women_std = [3,5,2,3,3]
width=0.35
fig, ax = plt.subplots()
fig
ax.bar(labels, men_means, width, yerr=men_std, label='Men')
ax.bar(labels, women_means, width, yerr=women_std, label='Women')
ax.set_ylabel('Scores')
ax.set_title('Scores by Group and Gender')
ax.legend()
plt.show()
#saving the chart
fig.savefig('barchart.jpeg')

#creating multiple plots ont eh same plotting area
import numpy as np
X= np.arange(0, 360)
X
y = np.sin(X)
y

#a figue with just one plot
fig, ax = plt.subplots()
ax.plot(X,y)

#Create two subplots 
f, (ax1, ax2)= plt.subplots(1,2, sharey=True)
ax1.plot(X,y)
ax1.set_title('Sharing Y axis')
ax2.scatter(X,y)


f, (ax1, ax2)= plt.subplots(2,1, sharex=True)
ax1.plot(X,y)
ax1.set_title('Sharing x axis')
ax2.scatter(X,y)


#Scatter plot
x = [5,7,8,7,2,17,2,9,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85]
plt.scatter(x,y)
plt.show()


from pydataset import data
mtcars = data('mtcars')
df = mtcars
df.columns

plt.scatter(x='wt', y='mpg', data=df)
plt.show()


#SUing seaborn

import seaborn as sns
sns.set()

tips_df = sns.load_dataset('tips')

tips_df

tips_df.columns
tips_df.total_bill

total_bill = tips_df.total_bill.to_numpy()
total_bill

tip = tips_df.tip.to_numpy()
plt.scatter(total_bill, tip)
plt.show()


#Drawing Heatmap
import seaborn as sns
import numpy as np
import pandas as pd

uniform_data = np.random.rand(10,10)
uniform_data
df = pd.DataFrame(uniform_data)
sns.heatmap(df)

#Plotting a histogram
from pydataset import data
mtcars = data('mtcars')
mtcars.head()
# continuous variables in the data
mtcars[['mpg', 'wt', 'hp' , 'disp']].head()
mtcars['mpg']

import matplotlib.pyplot as plt
plt.hist(mtcars.mpg)

plt.hist(mtcars.mpg, alpha=0.4, color='red') #alpha lightens the color

plt.hist(mtcars.wt, bins=5)

#plt.hist(np.histogram(mtcars.wt, bins=10)) NOT TO USE THIS
#continuous data- Regressor models, Discrete data- classification model
np.histogram(mtcars.wt) # identify the values of the histogram

#SNS distribution plot(distribution, frewuencu and nrmalized line)
sns.distplot(mtcars['mpg'], kde=True)

data_2 = np.random.normal(90,20,200)
data_2

sns.distplot(data_2, kde=True)


x = [1,2,3,4,5,6,7,8]
y = [3,9,6,5,8,8,2,3]

plt.scatter(x,y, color='r', marker='*')
plt.scatter(y,x, color='b', marker='o')

size = (30*np.random.rand(100))
size

plt.scatter(x,y,c='r', marker='o', s=size)
#size of plots denotes 3rd dimension


#PIE CHARt

sizes = [60,30,10]
labels = ['BBA', 'MBA', 'PHD']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
ax1.axis('equal')
plt.show()


#lambda function:
f_lambda = lambda x: x**2
a= f_lambda(3)
a

#map 
squared = []
items=[1,2,3,4,5]
for i in items:
    squared.append(i**2)
    
squared

#map allows us to implemnet this in a simpler way

squared1 = list(map(lambda x: x**2, items))
squared1

li = [5,7,22,97,54,62,77,23,73,61]
final_list = list(map(lambda x: x*2, li))
final_list

#filter function returns the values
number_list = range(-5,5)
less_zer = list(filter(lambda x:x<0, number_list))
less_zer

#reduce
from functools import reduce
li = [5,8,10,20,50,100]
sum = reduce((lambda x,y: x+y), li)
print(sum)

#Base of advanced analytics

mu, sigma = 65, 100
s=np.random.normal(mu, sigma,100)
s
#ignored - barcontainer