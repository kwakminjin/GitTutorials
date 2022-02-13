#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing the basic librarires fot analysis

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")  #using style ggplot

get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.graph_objects as go
import plotly.express as px


# In[3]:


#Importing the dataset

df=pd.read_csv("c:/WineQT.csv")


# In[4]:


# looking the data set
df.head()


# In[5]:


#print the shape dataset
print("Shape The DataSet ", df.shape )


# In[6]:


#Checking the dtypes of all the columns

df.info()


# In[7]:


#checking null value 
df.isna().sum()


# In[ ]:


# No missing value


# In[8]:


# Describe value data set
df.describe().round(2)


# In[9]:


# Drop columns ID , because we don't need it.

df.drop(columns="Id",inplace=True)


# In[10]:


#the unique quality 

print("The Value Quality ",df["quality"].unique())


# In[11]:


#graph all the data set - just for looking
df.plot(figsize=(15,7))


# In[13]:


# making Group by 

ave_qu = df.groupby("quality").mean()
ave_qu


# In[14]:


# graph the group by

ave_qu.plot(kind="bar",figsize=(20,10))


# In[ ]:


# now we see  the effect of the elements on the quality


# In[15]:


# let see effect some of elements on the quality - details
plt.figure(figsize=(20,7))
sns.lineplot(data=df, x="quality",y="volatile acidity",label="Volatile Acidity")
sns.lineplot(data=df, x="quality",y="citric acid",label="Citric Acid")
sns.lineplot(data=df, x="quality",y="chlorides",label="chlorides")
sns.lineplot(data=df, x="quality",y="pH",label="PH")
sns.lineplot(data=df, x="quality",y="sulphates",label="Sulphates")
plt.ylabel("Quantity")
plt.title("Impact on quality")
plt.legend()
plt.show()


# In[ ]:


# we see no high effect this elements on the quality


# In[16]:


# effect the Alcohol in the quality

plt.figure(figsize=(15,7))
sns.lineplot(data=df, x="quality",y="alcohol")


# In[17]:


# effect the total sulfur dioxide in the quality
plt.figure(figsize=(15,7))
sns.lineplot(data=df, x="quality",y="total sulfur dioxide",color="b")


# In[18]:


# effect the free sulfur dioxide in the quality

plt.figure(figsize=(15,7))
sns.lineplot(data=df, x="quality",y="free sulfur dioxide",color="g")


# In[19]:


# using graph interactive the show the effect free and total - sulfur dioxide in the quality

px.scatter(df, x="free sulfur dioxide", y="total sulfur dioxide",animation_frame="quality")

