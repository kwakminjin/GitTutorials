#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[37]:


df=pd.read_csv("c:/Life Expectancy Data.csv")


# In[38]:


df.head()


# In[39]:


df.dtypes


# In[40]:


#Analyzing the data
df.describe()


# In[41]:


#Renaming some columns
df.rename({'Life expectancy ':'Life expectancy',' BMI ':'BMI',' HIV/AIDS':'HIV/AIDS',' thinness  1-19 years':'thinness  1-19 years',' thinness 5-9 years':'thinness 5-9 years'},axis=1,inplace=True)
df.rename({'Measles ':'Measles','under-five deaths ':'under-five deaths','Diphtheria ':'Diphtheria',})


# In[42]:


#FINDING RELATION OF NULL VALUE 


# In[43]:


list_of_null_values=dict(df.isnull().sum())


# In[44]:


null_value_cols=[]
for key,value in list_of_null_values.items():
    if value>0:
        null_value_cols.append(key)


# In[45]:


null_value_cols=null_value_cols[1:]


# In[46]:


plt.figure(figsize=(20,20))

for i in range(len(null_value_cols)):
    if i<12:
        plt.subplot(5,3,i+1)
        plt.scatter(df[null_value_cols[i]],df['Life expectancy'])
        plt.xlabel(null_value_cols[i])
        plt.ylabel('Life Expectancy')
    else:
        plt.subplot(5,3,i+1)
        plt.scatter(df[null_value_cols[i]],df['Life expectancy'])
        plt.xlabel(null_value_cols[i])
        plt.ylabel('Life Expectancy')
        
    

plt.tight_layout()


# In[47]:


#Identifying relationship between Country Status and Life Expectancy 
plt.figure(figsize=(10,10))
df.groupby(['Status'])['Life expectancy'].mean().plot(kind='bar')


# In[50]:


df.columns


# In[51]:


plt.figure(figsize=(10,10))
plt.scatter(df['Schooling'],df['Life expectancy'],c='pink')
plt.xlabel('Schooling')
plt.ylabel('Life Expectancy')


# In[52]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True)


# In[53]:


plt.figure(figsize=(50,15))
val=df.groupby('Country')['Life expectancy'].mean().sort_values(ascending=False).plot(kind='bar',fontsize=15)


# In[54]:


plt.figure(figsize=(50,15))
val=df.groupby('Country')['Adult Mortality'].mean().sort_values(ascending=False).plot(kind='bar',fontsize=15)


# In[55]:


plt.figure(figsize=(50,15))
val=df.groupby('Country')['infant deaths'].mean().sort_values(ascending=False).plot(kind='bar',fontsize=15)


# In[56]:


plt.figure(figsize=(50,15))
val=df.groupby('Country')['GDP'].mean().sort_values(ascending=False).plot(kind='bar',fontsize=15)


# In[57]:


plt.figure(figsize=(50,15))
val=df.groupby('Schooling')['GDP'].mean().sort_values(ascending=False).plot(kind='bar',fontsize=15)


# In[58]:


#HANDLING NULL VALUES
data=df.copy()


# In[59]:


data.isnull().sum()


# In[60]:


list_of_null_values=dict(data.isnull().sum())


# In[61]:


data.isnull().sum()


# In[ ]:





# In[ ]:




