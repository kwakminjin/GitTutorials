#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_validate,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report

#from imblearn.over_sampling import SMOTE


# In[5]:


df = pd.read_csv("c:/bank-additional-full.csv",sep=";")


# In[9]:


df = df.rename(columns=str.lower)
#str.lower()은 str의 문자 중, 대문자를 소문자로 변경해줍니다.

for column in df.select_dtypes(include="object").columns.tolist():
    df[column] = df[column].apply(lambda x: x.lower()) 


# In[10]:


df.head()


# In[11]:


df.shape


# In[12]:


df.info()


# In[13]:


df.isna().sum()


# In[14]:


categorical_features = df.select_dtypes(include="object").drop(columns="y").columns.tolist()
numerical_features = df.select_dtypes(exclude="object").columns.tolist()


# In[17]:


categorical_features


# In[18]:


fig,ax = plt.subplots(nrows=5,ncols=2,figsize=(20,20))

for i in range(5):
    for j in range(2):
        sns.countplot(x=categorical_features[2*i+j],data=df,palette="viridis",ax=ax[i,j])


# In[19]:


df["job"].value_counts()


# In[20]:


df["education"].value_counts()


# In[21]:


sns.countplot(x="y",data=df,palette="viridis")

plt.show()


# In[22]:


fig,ax = plt.subplots(nrows=5,ncols=2,figsize=(20,20))

for i in range(5):
    for j in range(2):
        sns.kdeplot(x=numerical_features[2*i+j],data=df,hue="y",palette="viridis",fill=True,ax=ax[i,j])


# In[23]:


df[categorical_features] = OrdinalEncoder().fit_transform(df[categorical_features])


# In[24]:


df[numerical_features] = StandardScaler().fit_transform(df[numerical_features])


# In[27]:


df["y"] = LabelEncoder().fit_transform(df["y"])


# In[29]:


plt.figure(figsize=(20,20))

sns.heatmap(df.corr(),cmap="viridis",annot=True,fmt=".0%",square=True)

plt.show()
#cmap인자를 이용하여 히트 맵 색상을 바꿀 수 있음


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




