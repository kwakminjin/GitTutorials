#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Importing the basic librarires fot analysis

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")  #using style ggplot

get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.graph_objects as go
import plotly.express as px


# In[4]:


#Importing the dataset

df=pd.read_csv("c:/WineQT.csv")


# In[5]:


# looking the data set
df.head()


# In[6]:


#print the shape dataset
print("Shape The DataSet ", df.shape )


# In[8]:


#Checking the dtypes of all the columns

df.info()


# In[9]:


#checking null value 
df.isna().sum()


# In[ ]:


# No missing value


# In[10]:


# Describe value data set
df.describe().round(2)


# In[11]:


# Drop columns ID , because we don't need it.

df.drop(columns="Id",inplace=True)


# In[12]:


#the unique quality 

print("The Value Quality ",df["quality"].unique())


# In[13]:


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


# In[1]:


#Importing the basic librarires for building model


from sklearn.linear_model import LinearRegression ,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error ,mean_squared_error, median_absolute_error,confusion_matrix,accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC ,SVR


# In[14]:


#Defined X value and y value , and split the data train

X = df.drop(columns="quality")           
y = df["quality"]    # y = quality


# In[15]:


# split the data train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("X Train : ", X_train.shape)
print("X Test  : ", X_test.shape)
print("Y Train : ", y_train.shape)
print("Y Test  : ", y_test.shape)


# In[16]:


# using the model LinearRegression
LR_model=LinearRegression()

# fit model
LR_model.fit(X_train,y_train)


# In[17]:


# Score X and Y - test and train

print("Score the X-train with Y-train is : ", LR_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", LR_model.score(X_test,y_test))

# Expected value Y using X test
y_pred_LR=LR_model.predict(X_test)

# Model Evaluation
print( " Model Evaluation Linear R : mean absolute error is ", mean_absolute_error(y_test,y_pred_LR))
print(" Model Evaluation Linear R : mean squared  error is " , mean_squared_error(y_test,y_pred_LR))
print(" Model Evaluation Linear R : median absolute error is " ,median_absolute_error(y_test,y_pred_LR)) 


# In[18]:


# using the model Logistic Regression

Lo_model=LogisticRegression(solver='liblinear') #최적화에 사용할 알고리즘 결정

# fit model

Lo_model.fit(X_train,y_train)


# Score X and Y - test and train model Logistic Regression

print("Score the X-train with Y-train is : ", Lo_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", Lo_model.score(X_test,y_test))

# Expected value Y using X test
y_pred_Lo=Lo_model.predict(X_test)

# Model Evaluation
print( " Model Evaluation Logistic R : mean absolute error is ", mean_absolute_error(y_test,y_pred_Lo))
print(" Model Evaluation Logistic R : mean squared  error is " , mean_squared_error(y_test,y_pred_Lo))
print(" Model Evaluation Logistic R : median absolute error is " ,median_absolute_error(y_test,y_pred_Lo)) 

print(" Model Evaluation Logistic R : accuracy score " , accuracy_score(y_test,y_pred_Lo))


# In[19]:


# using the model Decision Tree Classifier
Tree_model=DecisionTreeClassifier(max_depth=10)
# fit model
Tree_model.fit(X_train,y_train)

# Score X and Y - test and train

print("Score the X-train with Y-train is : ", Tree_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", Tree_model.score(X_test,y_test))


# In[20]:


print("The Important columns \n",Tree_model.feature_importances_) #변수 중요도


# In[21]:


df.head(0)


# In[22]:


print("The classes ",Tree_model.classes_) #.classes_ 속성은 0번부터 순서대로 변환된 인코딩 값에 대한 원본값을 가지고 있음

y_pred_T =Tree_model.predict(X_test)

print(" Model Evaluation Decision Tree : accuracy score " , accuracy_score(y_test,y_pred_T))


# In[24]:


# using the model SVC
#SVM(Support Vector Machine) 분류


svc_model=SVC(C=50,kernel="rbf")
## SVM, kernel = 'rbf'로 비선형분리 진행


# fit model
svc_model.fit(X_train,y_train)

y_pred_svc =svc_model.predict(X_test)

print("Score the X-train with Y-train is : ", svc_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", svc_model.score(X_test,y_test))
print(" Model Evaluation Decision Tree : accuracy score " , accuracy_score(y_test,y_pred_svc))


# In[25]:


# using the model SVR
#서포트 백터 머신(from sklearn.svm import SVR)

svr_model=SVR(degree=1,coef0=1, tol=0.001, C=1.5,epsilon=0.001)
#1차항으로 설정, degree = 1
#coef0는 모델이 높은 차수와 낮은 차수에 얼마나 영향을 받을지 조절
#하이퍼 파라미터 C epsilon

# fit model
svr_model.fit(X_train,y_train)

y_pred_svr =svc_model.predict(X_test)

print("Score the X-train with Y-train is : ", svr_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", svr_model.score(X_test,y_test))
print(" Model Evaluation Decision Tree : accuracy score " , accuracy_score(y_test,y_pred_svr))


# In[26]:


# using the model K Neighbors Classifier

K_model = KNeighborsClassifier(n_neighbors = 5)
K_model.fit(X_train, y_train)

y_pred_k = K_model.predict(X_test)

print("Score the X-train with Y-train is : ", K_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", K_model.score(X_test,y_test))
print(" Model Evaluation K Neighbors Classifier : accuracy score " , accuracy_score(y_test,y_pred_k))

