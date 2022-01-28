#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


data = pd.read_csv("c:\winequality-red.csv")


# In[6]:


data.head()


# In[7]:


data.corr


# In[8]:


data.columns


# In[9]:


data.info()

#fixed.acidity(결합산) : 와인의 산도를 제어한다. 

#volatile.acidity(휘발산) : 와인의 향에 연관이 많다. 

#citric.acid(구연산) : 와인의 신선함을 유지시켜주는 역할을 하며, 산성화에 연관을 미친다. 

#residual.sugar(잔여 설탕) : 와인의 단맛을 올려준다. 

#chlorides(염소) : 와인의 짠맛과 신맛을 좌우하는 성분이다. 

#free.sulfur.dioxide / total.sulfur.dioxide / sulphates(황 화합물) : 특정 박테리아와 효모를 죽여 와인의 보관도를 높여준다. 

#density(밀도) : 바디의 높고 낮음을 표현하는 와인의 바디감을 의미한다. 

#pH(산성도) : 와인의 신맛의 정도를 나타낸다. 

#alcohol(알코올) : 와인에 단맛을 주며 바디감에 영향을 준다. 

#quality(퀄리티) : 결과적으로 다른 변수들을 이용하여 예측하려고 하는 변수로 와인의 퀄리티를 나타낸다. 


# In[14]:


data['quality'].unique() #유일값들


# In[15]:


#count of each target variable
from collections import Counter
Counter(data['quality'])


# In[16]:


#count of the target variable
sns.countplot(x='quality', data=data)


# In[19]:


#Plot a boxplot to check for Outliers
#Target variable is Quality. So will plot a boxplot each column against target variable
sns.boxplot('quality', 'fixed acidity', data = data)


# In[22]:


sns.boxplot('quality', 'volatile acidity', data = data)


# In[23]:


sns.boxplot('quality', 'citric acid', data = data)


# In[24]:


sns.boxplot('quality', 'residual sugar', data = data)


# In[25]:


sns.boxplot('quality', 'chlorides', data = data)


# In[26]:


sns.boxplot('quality', 'free sulfur dioxide', data = data)


# In[27]:


sns.boxplot('quality', 'total sulfur dioxide', data = data)


# In[28]:


sns.boxplot('quality', 'density', data = data)


# In[29]:


sns.boxplot('quality', 'pH', data = data)


# In[30]:


sns.boxplot('quality', 'sulphates', data = data)


# In[31]:


sns.boxplot('quality', 'alcohol', data = data)


# In[32]:


#boxplots show many outliers for quite a few columns. Describe the dataset to get a better idea on what's happening
data.describe()
#fixed acidity - 25% - 7.1 and 50% - 7.9. Not much of a variance. Could explain the huge number of outliers
#volatile acididty - similar reasoning
#citric acid - seems to be somewhat uniformly distributed
#residual sugar - min - 0.9, max - 15!! Waaaaay too much difference. Could explain the outliers.
#chlorides - same as residual sugar. Min - 0.012, max - 0.611
#free sulfur dioxide, total suflur dioxide - same explanation as above


# In[33]:


#next we shall create a new column called Review. This column will contain the values of 1,2, and 3. 
#1 - Bad
#2 - Average
#3 - Excellent
#This will be split in the following way. 
#1,2,3 --> Bad
#4,5,6,7 --> Average
#8,9,10 --> Excellent
#Create an empty list called Reviews
reviews = []
for i in data['quality']:
    if i >= 1 and i <= 3:
        reviews.append('1')
    elif i >= 4 and i <= 7:
        reviews.append('2')
    elif i >= 8 and i <= 10:
        reviews.append('3')
data['Reviews'] = reviews


# In[34]:


#view final data
data.columns


# In[35]:


data['Reviews'].unique()


# In[36]:


Counter(data['Reviews'])


# In[37]:


x = data.iloc[:,:11] # 데이터프레임의 행이나 컬럼에 인덱스 값으로 접근.
y = data['Reviews']


# In[38]:


x.head(10)


# In[46]:


y.head(10)


# In[47]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)


# In[48]:


#view the scaled features
print(x)


# In[49]:


from sklearn.decomposition import PCA
pca = PCA()
x_pca = pca.fit_transform(x)


# In[50]:


#plot the graph to find the principal components
plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
#배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수
#주성분 퍼센트
plt.grid()


# In[52]:


#AS per the graph, we can see that 8 principal components attribute for 90% of variation in the data. 
#we shall pick the first 8 components for our prediction.
pca_new = PCA(n_components=8)
x_new = pca_new.fit_transform(x)


# In[53]:


print(x_new)


# In[55]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.25)


# In[56]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[57]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_predict = lr.predict(x_test)


# In[58]:


#print confusion matrix and accuracy score
lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_acc_score = accuracy_score(y_test, lr_predict)
print(lr_conf_matrix)
print(lr_acc_score*100)


# In[59]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_predict = dt.predict(x_test)


# In[60]:


#print confusion matrix and accuracy score
dt_conf_matrix = confusion_matrix(y_test, dt_predict)
dt_acc_score = accuracy_score(y_test, dt_predict)
print(dt_conf_matrix)
print(dt_acc_score*100)


# In[61]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
nb_predict=nb.predict(x_test)


# In[62]:


#print confusion matrix and accuracy score
nb_conf_matrix = confusion_matrix(y_test, nb_predict)
nb_acc_score = accuracy_score(y_test, nb_predict)
print(nb_conf_matrix)
print(nb_acc_score*100)


# In[63]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_predict=rf.predict(x_test)


# In[64]:


#print confusion matrix and accuracy score
rf_conf_matrix = confusion_matrix(y_test, rf_predict)
rf_acc_score = accuracy_score(y_test, rf_predict)
print(rf_conf_matrix)
print(rf_acc_score*100)


# In[65]:


from sklearn.svm import SVC


# In[74]:


#we shall use the rbf kernel first and check the accuracy
lin_svc = SVC()
lin_svc.fit(x_train, y_train)
lin_predict=lin_svc.predict(x_test)


# In[75]:


#print confusion matrix and accuracy score
lin_svc_conf_matrix = confusion_matrix(y_test, lin_predict)
lin_svc_acc_score = accuracy_score(y_test, lin_predict)
print(lin_svc_conf_matrix)
print(lin_svc_acc_score*100)


# In[76]:


rbf_svc = SVC(kernel='linear')
rbf_svc.fit(x_train, y_train)
rbf_predict=rbf_svc.predict(x_test)


# In[77]:


rbf_svc_conf_matrix = confusion_matrix(y_test, rbf_predict)
rbf_svc_acc_score = accuracy_score(y_test, rbf_predict)
print(rbf_svc_conf_matrix)
print(rbf_svc_acc_score*100)

