#!/usr/bin/env python
# coding: utf-8

# In[17]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[18]:


df=pd.read_csv('c:/Salary Dataset.csv')
df.head()


# In[19]:


df.info()


# In[20]:


df=df.dropna()


# In[21]:


df["Job Title"].unique()


# In[22]:


df["Job Title"].replace({"Software Engineer - Machine Learning": "Machine Learning Software Engineer"}, inplace=True)
df["Job Title"].replace({"Machine Learning Data Associate": "Machine Learning Associate","Associate Machine Learning Engineer":"Machine Learning Associate"}, inplace=True)
df["Job Title"].replace({"Data Science": "Data Scientist"}, inplace=True)
df["Job Title"].unique()


# In[23]:


df["Company Name"].unique()


# In[24]:


df["Location"].unique()


# In[25]:


df["Salary"]


# In[27]:


df[['Salary', 'Duration']] = df['Salary'].str.split('/', 1, expand=True) #expand=True이면 여러 컬럼, False이면 1개 컬럼에 리스트
df["Duration"].unique()


# In[28]:


import re
df["Salary"]=df["Salary"].map(lambda x: re.sub(r'\W+', '', x)) ## 단어 단위로 출력
#re.sub('패턴', 바꿀문자열', '문자열', 바꿀횟수)

#\W : 영문자 및 _ 문자를 제외한 문자와 일치

##정규식 일치부를 문자열에서 제거
#치환 텍스트에 정규식 일치부 삽입


# In[30]:


df["Salary"]=pd.to_numeric(df["Salary"], errors='coerce') #문자열을 강제로 NaN으로 바꾸면서
df["Salary"]


# In[31]:


df['Salaries Reported'].unique()


# In[32]:


import seaborn as sns


# In[33]:


sns.set_style('white')
sns.set_context("paper", font_scale = 2)
sns.displot(data=df, x="Salaries Reported", kind="hist", bins = 100, aspect = 1.5)
df["Salaries Reported"].mean()


# In[34]:


df.head()


# In[35]:


sal_per_year = df.copy()
sal_per_year = sal_per_year[sal_per_year['Duration'] == "yr"]


# In[36]:


sal_per_year = sal_per_year.sort_values(by=['Salary'], ascending=False)


# In[37]:


sal_per_year[:5].plot.bar(x='Company Name', y='Salary')

