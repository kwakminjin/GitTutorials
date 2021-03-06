#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train = pd.read_csv('C:/train.csv')
test = pd.read_csv('C:/test.csv')


# In[3]:


print(train.shape)
print(test.shape)


# In[4]:


train.head()


# In[5]:


sns.heatmap(train.isnull(), yticklabels = False, cbar = False)

#we can average fill the missing fields for age. 
#but cabin seem to have too many missing information, we might need to drop the column.


# In[6]:


train.info()


# In[7]:


train.describe()

#a few columns i'm interested to explore if it affects survivability is:
    #fare 
    #sibsp
    #parch
    #age


# In[8]:


sns.displot(data = train, x = 'Age')


# In[9]:


sns.displot(data = train, x = 'Fare', bins = 10)


# In[ ]:


sns.displot(data = train, x = 'SibSp', bins = 10)


# In[10]:


sns.displot(data = train, x = 'Parch')


# In[11]:


pd.pivot_table(data = train, index = 'Survived', values = ['Age', 'Fare', 'SibSp', 'Parch'])


# In[12]:


#Preliminary look at the data we deduce: 
    #Age is normally distributed. 
    #Bulk of the passengers paid for a cheaper ticket. 
    #Bulk of the passengers don't have any siblings or spouse. 
    #Bulk of the passengers don't have parent or child.


# In[13]:


sns.heatmap(train.isnull(), yticklabels = False, cbar = False)

#fill the missing fields for age with average.
#but cabin seem to have too many missing information, we to drop the column.


# In[15]:


train['Age'].fillna(train['Age'].mean(), inplace = True)


# In[16]:


train.drop('Cabin', axis = 1, inplace = True)


# In[17]:


sns.heatmap(train.isnull(), yticklabels = False, cbar = False)


# In[18]:


train.head()


# In[19]:


#lets see if the title of passengers affect survivability

train['title'] = train.Name.apply(lambda x : x.split(',')[1].split('.')[0].strip())

#strip() ?????? : ????????? ????????? ?????? ?????? ????????? ?????? ??????


# In[20]:


train.head(1)


# In[24]:


pd.pivot_table(data = train, index = 'Survived', columns = 'title', values = 'Age', aggfunc = 'count')


# In[ ]:


#the captain of the ship did not survive. We can assume that he tried his best to get passengers off before himself.
#col, major, sir - military personnels are more likely to survive.
#doctors are also very likely to survive (3/4)
#the only countess onboard survived. She was later praised for helping row the safety craft to a rescue ship.
#most passengers fall into the Miss, Mr, and Mrs categories. Of which, male title (Mr) is the most likely not to survive

#so yes, title has a correlation to survivability


# In[25]:


#lets see if the class of passengers affect survivability

sns.countplot(data = train, x = 'Survived', hue = 'Pclass')

#3rd class passengers will most likely not survive.


# In[27]:


#lets see if the age of passengers affect survivability

bins = [18, 30, 40, 50, 60, 70, 120]
labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']
train['age_range'] = pd.cut(train.Age.astype(np.int64), bins, labels = labels, include_lowest = True)


# In[28]:


sns.countplot(data = train, x = 'Survived', hue = 'age_range')


# In[29]:


train.age_range.value_counts()


# In[ ]:


#in both categories: survived and did not survive, the majority are aged 18-29.
#in both categories: slowly trickles down in count as the age range increases.
#this is becos most passengers are aged 18-29. Naturally, this would result in both categories having the same distribution.
#no conclusive result can be derived.
#not used.


# In[30]:


train.head()

