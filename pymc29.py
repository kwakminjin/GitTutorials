#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np

# data visualization library 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context='notebook', style='darkgrid', palette='colorblind', font='sans-serif', font_scale=1, rc=None)
matplotlib.rcParams['figure.figsize'] =[8,8]
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['font.family'] = 'sans-serif'


# In[53]:


# dataprep
from dataprep.eda import *
from dataprep.eda.missing import plot_missing
from dataprep.eda import plot_correlation


# In[54]:


covid = pd.read_csv('c:/Covid Dataset.csv')
covid


# In[55]:


covid.info()


# In[56]:


covid.describe(include='all')


# In[57]:


covid.columns


# In[58]:


plot_missing(covid)


# In[59]:


# create a table with data missing 
missing_values=covid.isnull().sum() # missing values

percent_missing = covid.isnull().sum()/covid.shape[0]*100 # missing value %

value = {
    'missing_values ':missing_values,
    'percent_missing %':percent_missing  
}
frame=pd.DataFrame(value)
frame


# In[60]:


sns.countplot(x='COVID-19',data=covid)


# In[61]:


covid["COVID-19"].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',shadow=True)
plt.title('number of cases');


# In[62]:


sns.countplot(x='Breathing Problem',data=covid)


# In[63]:


sns.countplot(x='Breathing Problem',hue='COVID-19',data=covid)


# In[64]:


sns.countplot(x='Fever',hue='COVID-19',data=covid);


# In[65]:


sns.countplot(x='Dry Cough',hue='COVID-19',data=covid)


# In[66]:


sns.countplot(x='Sore throat',hue='COVID-19',data=covid)


# In[67]:


from sklearn.preprocessing import LabelEncoder
e=LabelEncoder()
#Categorical 데이터를 Numerical 로 변환


# In[68]:


covid['Breathing Problem']=e.fit_transform(covid['Breathing Problem'])
covid['Fever']=e.fit_transform(covid['Fever'])
covid['Dry Cough']=e.fit_transform(covid['Dry Cough'])
covid['Sore throat']=e.fit_transform(covid['Sore throat'])
covid['Running Nose']=e.fit_transform(covid['Running Nose'])
covid['Asthma']=e.fit_transform(covid['Asthma'])
covid['Chronic Lung Disease']=e.fit_transform(covid['Chronic Lung Disease'])
covid['Headache']=e.fit_transform(covid['Headache'])
covid['Heart Disease']=e.fit_transform(covid['Heart Disease'])
covid['Diabetes']=e.fit_transform(covid['Diabetes'])
covid['Hyper Tension']=e.fit_transform(covid['Hyper Tension'])
covid['Abroad travel']=e.fit_transform(covid['Abroad travel'])
covid['Contact with COVID Patient']=e.fit_transform(covid['Contact with COVID Patient'])
covid['Attended Large Gathering']=e.fit_transform(covid['Attended Large Gathering'])
covid['Visited Public Exposed Places']=e.fit_transform(covid['Visited Public Exposed Places'])
covid['Family working in Public Exposed Places']=e.fit_transform(covid['Family working in Public Exposed Places'])
covid['Wearing Masks']=e.fit_transform(covid['Wearing Masks'])
covid['Sanitization from Market']=e.fit_transform(covid['Sanitization from Market'])
covid['COVID-19']=e.fit_transform(covid['COVID-19'])
covid['Dry Cough']=e.fit_transform(covid['Dry Cough'])
covid['Sore throat']=e.fit_transform(covid['Sore throat'])
covid['Gastrointestinal ']=e.fit_transform(covid['Gastrointestinal '])
covid['Fatigue ']=e.fit_transform(covid['Fatigue '])


# In[69]:


covid.head()


# In[70]:


covid.dtypes.value_counts()


# In[71]:


covid.describe(include='all')


# In[72]:


covid.hist(figsize=(20,15));


# In[73]:


plot_correlation(covid) #결과가 왜다른지는 잘모르겠다.


# In[ ]:


corr=covid.corr()
corr.style.background_gradient(cmap='coolwarm',axis=None)


# In[74]:


corr=covid.corr()
corr.style.background_gradient(cmap='coolwarm',axis=None)

