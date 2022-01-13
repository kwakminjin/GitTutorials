#!/usr/bin/env python
# coding: utf-8

# In[18]:


##1.선형회귀
# Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[19]:


#Read csv file :

data = pd.read_csv('c:\Fuel.csv')
data.head()


# In[20]:


#Let's select some features to explore more :
#여기서 우리의 목표는 데이터 세트의 "엔진 크기"값에서 "co2 배출량"값을 예측하는 것


data = data[["ENGINESIZE", "CO2EMISSIONS"]]


# In[24]:


# ENGINESIZE vs CO2EMISSIONS:
#데이터를 산점도로 시각화

plt.scatter(data["ENGINESIZE"] , data["CO2EMISSIONS"] , color="blue")
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()


# In[25]:


# Generating training and testing data from our data:
# We are using 80% data for training.
#데이터를 훈련 및 테스트데이터로 나누기

#모델의 정확성을 확인하기 위해 데이터를 훈련 및 테스트 데이터 세트로 나눌 것입니다. 훈련 데이터를 사용하여 모델을 훈련 한 다음 테스트 데이터 세트를 사용하여 모델의 정확성을 확인합니다.

train = data[:(int((len(data)*0.8)))]
test = data[(int((len(data)*0.8))):]


# In[29]:


# Modeling:
# Using sklearn package to model data :
#모델훈련
#모델을 훈련시키고 최적 회귀선에 대한 계수를 찾는 방법은 다음과 같습니다.

regr = linear_model.LinearRegression()
train_x = np.array(train[["ENGINESIZE"]])
train_y = np.array(train[["CO2EMISSIONS"]])
regr.fit(train_x,train_y)

# The coefficients:
print ("coefficients : ",regr.coef_) #Slope
print ("Intercept : ",regr.intercept_) #Intercept


# In[31]:


# Plotting the regression line:
#최적선을 플로팅
#계수를 기반으로 데이터 세트에 가장 적합한 선을 그릴 수 있습니다.

plt.scatter(train["ENGINESIZE"], train["CO2EMISSIONS"], color="blue")
plt.plot(train_x, regr.coef_*train_x + regr.intercept_, "-r")
plt.xlabel("Engine size")
plt.ylabel("Emission")


# In[32]:


# Predicting values:
# Function for predicting future values :
#예측기능
#테스트 데이터 세트에 예측 기능을 사용할 것입니다.

def get_regression_predictions(input_features,intercept,slope):
    predicted_values = input_features*slope + intercept
    return predicted_values


# In[33]:


# Predicting emission for future car:
#이산화탄소 배출량 예측
#회귀선을 기반으로 CO2 배출량 예측.
my_engine_size = 3.5
estimatd_emission = get_regression_predictions(my_engine_size,regr.intercept_[0],regr.coef_[0][0])
print ("Estimated Emission :",estimatd_emission)


# In[35]:


# Checking various accuracy:
#테스트 데이터의 정확성 확인
#실제 값을 데이터 세트의 예측 값과 비교하여 모델의 정확성을 확인할 수 있습니다.

from sklearn.metrics import r2_score
test_x = np.array(test[["ENGINESIZE"]])
test_y = np.array(test[["CO2EMISSIONS"]])
test_y_ = regr.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Mean sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




