# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 11:04:52 2022

@author: Deep Khut
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv("C:/Users/jayes/Downloads/who_suicide_statistics.csv/who_suicide_statistics.csv")

# look at 1st 5 data points
data.head(5)

data.info()# print the concise summery of the dataset

data.shape # 43776 Rows, 6 Columns

# counts total row in each col. that have null values
# note: all the na columns are type Object
data.isna().sum()

# From above, we can see that, suicides_no & population, have null values.
#Lets, fill the null values with zero using 'fillna'
data= data.fillna(0)
# Now, we have no null columns!
data.isna().sum()

# The different age groups
data['age'].unique()

# lists the different countries
data['country'].unique()

# the Number of different Countries our dataset is from
data['country'].nunique()
# Our dataset is from 141 different Countries

# The different country groups
data['year'].unique()

# Replace 0 values with, NA
data['suicides_no'] = data['suicides_no'].replace(0,np.NAN)

# replace Na values with, mean value
mean_value=data['population'].mean()

data['population']=data['population'].fillna(mean_value)

# do same for Popualation
# replace Na values with, mean value
mean_value=data['suicides_no'].mean()

data['suicides_no']=data['suicides_no'].fillna(mean_value)


#   Exploratory Data Analysis
#   Which year has the most Suicides ? Which year has the least Suicides ?

data['suicides_no'] = data['suicides_no'].replace(0,np.NAN)

mean_value=data['suicides_no'].mean()
data['suicides_no']=data['suicides_no'].fillna(mean_value)

def find_minmax(x):
    #use the function 'idmin' to find the index of lowest suicide
    min_index = data[x].idxmin()
    #use the function 'idmax' to find the index of Highest suicide
    high_index = data[x].idxmax()
    
    high = pd.DataFrame(data.loc[high_index,:])
    low = pd.DataFrame(data.loc[min_index,:])
    
    #print the Year with high and low suicide
    print("Year Which Has Highest "+ x + " : ",data['year'][high_index])
    print("Year Which Has Lowest "+ x + "  : ",data['year'][min_index])
    return pd.concat([high,low],axis = 1)

find_minmax('suicides_no')


# year-wise analysis of mean number sucidies of each year
            # x             #y
data.groupby('year')['suicides_no'].mean().plot()

#setup the title and labels of the figure.
plt.title("Year vs. Suicide Count",fontsize = 14)
plt.xlabel('Year',fontsize = 13)
plt.ylabel('Suicide Count',fontsize = 13)

#setup the figure size.
sns.set(rc={'figure.figsize':(10,5)})
sns.set_style("whitegrid")

#   Research Question 2: Which country has the most Suicides ? Which country has the least Suicides ?
def find_minmax(x):
     #use the function 'idmin' to find the index of lowest suicide
    min_index = data[x].idxmin()
    #use the function 'idmax' to find the index of Highest suicide
    high_index = data[x].idxmax()
    
    high = pd.DataFrame(data.loc[high_index,:])
    low = pd.DataFrame(data.loc[min_index,:])
    
    #print the country with high and low suicide
    print("Country Which Has Highest "+ x + " : ",data['country'][high_index])
    print("Country Which Has Lowest "+ x + "  : ",data['country'][min_index])
    return pd.concat([low,high],axis = 1)

find_minmax('suicides_no')


#calculate mean of suicides_no col
meanSuicide = data['suicides_no'].mean()
#calculate mean of pop. col
meanPop = data['population'].mean()

#NOTE: You may replace NA values with mean, OR Drop them, I showed both
    
# drops any Na rows
data = data.dropna()    
    
# Replace 0 or NaN suicides_no, with the mean Suicide    
data['suicides_no'] = data['suicides_no'].replace(np.NAN,meanSuicide)

# Replace 0 or NaN populations, with the mean Populations
data['population'] = data['population'].replace(np.NAN,meanPop)
data['population'] = data['population'].replace(0,meanPop)

# peform operation
data['suicide_per_pop'] = data['suicides_no']/data['population']

        # another way of peforming the operation from above:
# data['suicide_per_pop'] = data.apply(lambda row: row.suicides_no / row.population, axis = 1) 

data.tail(3)

find_minmax('suicide_per_pop')


# year-wise analysis of mean number sucidies of each year
            # x             #y
data.groupby('country')['suicides_no'].mean().plot()

#info = pd.DataFrame(data['country'].sort_values(ascending = False))

#setup the title and labels of the figure.
plt.title("Country Vs. Suicide Count",fontsize = 14)
plt.xlabel('Country',fontsize = 13)
plt.ylabel('Suicide Count',fontsize = 13)

#setup the figure size.
sns.set(rc={'figure.figsize':(10,5)})
sns.set_style("whitegrid")


#     Research Question 3: Are certain age groups more inclined to suicide?
sample = data.sample(3)
sample
 # grabs first 2 chars from Age Column
data['AgeNum'] = data['age'].str[:2]

# remove all instances of dash -
data['AgeNum'] = data['AgeNum'].map(lambda x: x.replace('-',''))

# now, convert it to type int (not Object)
data['AgeNum'] = data['AgeNum'].astype(int)

data['AgeNum'].tail(3)


# creates Age Categories
def AgeGroup(x):
    if(x >= 60):
        return "Elderly"
    elif(x >= 30):
        return "Middle_Aged_Adults"
    elif(x >= 18):
        return "Adults"
    else:
        return "Adolescent"
# Map each row in the Col to the AgeGroup Method
data['AgeCategory'] = data['AgeNum'].map(lambda x: AgeGroup(x))
# convert it back to type String
data['AgeCategory'] = data['AgeCategory'].astype(str)
data['AgeCategory'].tail(3)

data['AgeNum'] .tail(3)

data.head(3)

sns.catplot(x="AgeCategory", y="suicides_no",palette="ch:.25", kind="bar",data=data);

plt.title('Age vs. Suicide Count',size=25)
plt.xlabel('Age Category',size=20)
plt.ylabel('Suicide Count',size=20)


#    Research Question 4: What is the relationship between the gender and the number of suicides?

# there is an equal number of Males & Females in our data
data['sex'].value_counts()

sns.catplot(x="sex", y="suicides_no", hue="AgeCategory", kind="bar", data=data);



#     Machine Learning + Predictive Analytics

data.head(3)
newData= data.loc[:,['year','sex','AgeNum','suicides_no']]
newData.head(3)
X = newData.iloc[:, :-1].values # grab the every col except last
y = newData.iloc[:, -1].values # grab last col

#Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)  
print(x_train)
print(x_test)
print(y_train)
print(y_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
# we are predicting the suicide count given certain demographics

# A 55 year old male, in 2001 
# suicide count of about 187.
print(regressor.predict([[1,0,2001,55]]))
