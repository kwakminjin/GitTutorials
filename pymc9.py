#!/usr/bin/env python
# coding: utf-8

# In[2]:


# data analysis and wrangling #데이터 분석 및 논쟁
import pandas as pd
import numpy as np
import random as rnd

# visualization #시각화
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning #머신러닝
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[4]:


train_df = pd.read_csv('c:/train.csv')
test_df = pd.read_csv('c:/test.csv')
combine = [train_df, test_df]


# In[5]:


print(train_df.columns.values)


# In[6]:


train_df.head()


# In[7]:


train_df.info()
print('_'*40)
test_df.info()


# In[8]:


train_df.describe()
# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
# Review Parch distribution using `percentiles=[.75, .8]`
# SibSp distribution `[.68, .69]`
# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`


# In[9]:


train_df.describe(include=['O']) #문자열(String) select_dtypes (ex: df.describe (include=['0']))


# In[12]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'],
                                         as_index=False).mean().sort_values(by='Survived',
                                                                            ascending=False)
#as_index=False 구문은 이 그룹을 인덱스로 지정할 것인지 여부(지정하면 pclass가 인덱스가됨)


# In[16]:


train_df[["Sex", "Survived"]].groupby(['Sex'],
                                      as_index=False).mean().sort_values(by='Survived',
                                                                         ascending=False)


# In[17]:


train_df[["SibSp", "Survived"]].groupby(['SibSp'],
                                        as_index=False).mean().sort_values(by='Survived',
                                                                           ascending=False)


# In[18]:


train_df[["Parch", "Survived"]].groupby(['Parch'],
                                        as_index=False).mean().sort_values(by='Survived',
                                                                           ascending=False)


# In[23]:


g = sns.FacetGrid(train_df, col='Survived') #패싯그리드
#다양한 범주형 값을 가지는 데이터를 시각화하는데 좋은 방법
#행, 열 방향으로 서로 다른 조건을 적용하여 여러 개의 서브 플롯 제작
#각 서브 플롯에 적용할 그래프 종류를 map() 메서드를 이용하여 그리드 객체에 전달

#1. FacetGrid에 데이터프레임과 구분할 row, col, hue 등을 전달해 객체 생성

#2. 객체(facet)의 map 메서드에 그릴 그래프의 종류와 종류에 맞는 컬럼 전달
#예시 - distplot의 경우 하나의 컬럼 // scatter의 경우 두개의 컬럼

g.map(plt.hist, 'Age', bins=20) #x값 age


# In[30]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
#컬럼엔 생존자수를, 로우엔 Pclass를 넣었다.
#두 컬럼 집단별 나이의 분포를 볼 수 있게 됐다.

 
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# In[33]:


# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# In[34]:


# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# In[35]:


print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
#shape 하면 해당 차원이 몇 차원인지 표시해줍니다.

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




