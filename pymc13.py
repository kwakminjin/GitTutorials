#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sklearn
print(sklearn.__version__)


# In[3]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[4]:


import pandas as pd

iris = load_iris()
iris


# In[5]:


#iris.data는 iris 데이터 세트에서 feature만으로 된 데이터를 numpy로 가지고 있음
iris_data = iris.data
iris_data


# In[6]:


#iris.target는 붓꽃 데이터 세트에서 레이블(결정값) 데이터를 numpy로 가지고 있음
iris_label = iris.target
print(type(iris_label))
print(iris_label.shape) #갯수
iris_label #정답데이터


# In[8]:


#붓꽃 데이터 세트를 자세히 보기 위해 DataFrame으로 변환
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
iris_df.head()


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label,
                                                   test_size=0.2, random_state=9)


# In[11]:


print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)


# In[14]:


dt_clf = DecisionTreeClassifier(random_state=11)

dt_clf.fit(X_train, y_train)


# In[17]:


pred = dt_clf.predict(X_test)


# In[18]:


print(len(pred))
pred


# In[19]:


iris.target_names


# In[20]:


from sklearn.metrics import accuracy_score
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test,pred)))

