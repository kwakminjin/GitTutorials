#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import all the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[2]:


#Define a dataset for semi-supervised learning and establish a baseline in performance on the dataset
data = pd.read_csv('c:/Iris.csv')
data.head()


# In[3]:


#Information on features
data.info()


# In[6]:


fig,ax = plt.subplots(1,2,figsize=(20,5))
data.Species.value_counts().plot(kind='pie',ax=ax[0],autopct='%1.1f%%',explode=[0.05,0.05,0.05],shadow=True,colors=['#334550','#394184','#6D83AA'])
sns.countplot(x=data.Species,ax=ax[1],palette=['#334550','#394184','#6D83AA']);


# In[7]:


fig,ax = plt.subplots(1,2,figsize=(20,6))
data[data.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='red',label='Setosa',ax=ax[0])
data[data.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',ax=ax[0],color='green',label='Versicolor')
data[data.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',ax=ax[0],color='black',label='Virginica')

data[data.Species=='Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='red',label='Setosa',ax=ax[1])
data[data.Species=='Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',ax=ax[1],color='green',label='Versicolor')
data[data.Species=='Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',ax=ax[1],color='black',label='Virginica');


# In[8]:


data.drop('Id',axis=1,inplace=True)
#Convert the labels into a numeric form so as to convert them into the machine-readable form
encoder = LabelEncoder()
data.Species = encoder.fit_transform(data.Species)
data.head()


# In[20]:


#Define how many samples should be Unlabeled
random_unlabeled_points = np.random.RandomState(0).rand(len(data.Species)) <= 0.5
 #np.random.RandomState(0)난수생성
#0부터 1 사이에서 균일한 확률 분포로 실수 난수를 생성
random_unlabeled_points


# In[21]:


#Develop a semi-supervised classification dataset
unlabeled = data.Species.values
#Create "no label" for unlabeled data
unlabeled[random_unlabeled_points] = -1
unlabeled


# In[25]:


#Define Model
label_spreading = LabelSpreading()
#Fit model on training dataset
label_spreading.fit(data.drop('Species',axis=1),unlabeled) #데이터 피팅
#After the model is fit, the estimated labels for the labeled and unlabeled data in the training dataset is available via the “transduction_” attribute
label_spreading.transduction_ #변환?


# In[26]:


#Calculate score for test set
label_spreading.score(data.drop('Species',axis=1),label_spreading.transduction_)


# In[27]:


x = data.drop('Species',axis=1)
y = label_spreading.transduction_
#Split the dataset into train and test datasets
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.4)


# In[28]:


#Define supervised learning model
rf_model = RandomForestClassifier()
#Fit supervised learning model on entire training dataset
rf_model.fit(xtrain,ytrain)
#Make predictions on hold out test set
predictions = rf_model.predict(xtest)
#Calculate score for test set
print('Random Forest Classifier Accuracy: {:.2f}'.format(accuracy_score(ytest,predictions)))


# In[29]:


svm = SVC(kernel='rbf',gamma=.10, C=1.0)
svm.fit(xtrain, ytrain)
predictions = svm.predict(xtest)
print('SVM Classifier Accuracy: {:.2f}'.format(accuracy_score(ytest,predictions)))

