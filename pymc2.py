#!/usr/bin/env python
# coding: utf-8

# In[14]:


# Import the required libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[15]:


# Read the CSV file:
data = pd.read_csv("c:\Fuel.csv")
data.head()


# In[16]:


# Consider features we want to work on:
#X는 우리가 고려할 입력 특성을 저장하고 Y는 출력 값을 저장합니다.
X = data[[ "ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_CITY","FUELCONSUMPTION_HWY", 
 "FUELCONSUMPTION_COMB","FUELCONSUMPTION_COMB_MPG"]]
Y = data["CO2EMISSIONS"]


# In[17]:


# Generating training and testing data from our data:
# We are using 80% data for training.
#데이터를 테스트 및 학습 데이터 세트로 나눕니다.
#여기서는 훈련에 80 % 데이터를 사용하고 테스트에 20 % 데이터를 사용할 것입니다.
train = data[:(int((len(data)*0.8)))]
test = data[(int((len(data)*0.8))):]


# In[18]:


#Modeling:
#Using sklearn package to model data :
#여기서 우리는 데이터의 80 %로 모델을 훈련시킬 것입니다.
regr = linear_model.LinearRegression()
train_x = np.array(train[[ "ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_CITY",
 "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB","FUELCONSUMPTION_COMB_MPG"]])
train_y = np.array(train["CO2EMISSIONS"])
regr.fit(train_x,train_y)
test_x = np.array(test[[ "ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_CITY",
 "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB","FUELCONSUMPTION_COMB_MPG"]])
test_y = np.array(test["CO2EMISSIONS"])


# In[19]:


# print the coefficient values:
#입력 특성의 계수를 찾으십시오.

#이제 어떤 기능이 출력 변수에 더 중요한 영향을 미치는지 알아야합니다. 이를 위해 계수 값을 인쇄 할 것입니다. 음의 계수는 출력에 역효과가 있음을 의미합니다. 즉, 해당 기능의 값이 증가하면 출력 값이 감소합니다.

coeff_data = pd.DataFrame(regr.coef_ , X.columns , columns=["Coefficients"])
coeff_data


# In[20]:


#Now let’s do prediction of data:
#값을 예측하십시오.


Y_pred = regr.predict(test_x)


# In[21]:


# Check accuracy:
#모델의 정확도 :
from sklearn.metrics import r2_score
R = r2_score(test_y , Y_pred)
print ("R² :",R)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




