#!/usr/bin/env python
# coding: utf-8

# In[1]:


###지수 회귀 공식 : Y = a + b * c^x
#where, Y = output
#x = input feature
#a = shift value
#b = y - intercpet
#c = base

# Import required libraries:
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# In[2]:


# Dataset values :
#데이터 포인트를 삽입합니다.

day = np.arange(0,8)
weight = np.array([251,209,157,129,103,81,66,49])


# In[3]:


# Exponential Function :
#지수 함수 알고리즘을 구현합니다.

def expo_func(x, a, b):
    return a * b ** x


# In[9]:


#popt :Optimal values for the parameters
#pcov :The estimated covariance of popt
#최적의 모수 및 공분산 적용 :
#여기서 우리는 최적의 매개 변수 값을 찾기 위해 curve_fit을 사용합니다. popt, pcov 라는 두 개의 변수를 반환합니다 .
#popt 는 최적 매개 변수 값을 저장하고 pcov는 공분산 값을 저장합니다. popt 변수에 두 개의 값이 있음을 알 수 있습니다. 이러한 값은 최적의 매개 변수입니다. 아래에 표시된대로 이러한 매개 변수를 사용하여 최적의 곡선을 플로팅합니다.

popt, pcov = curve_fit(expo_func, day, weight)
weight_pred = expo_func(day,popt[0],popt[1])
print(popt)
print(pcov)


# In[12]:


# Plotting the data
#데이터를 플로팅합니다.
#찾은 계수로 데이터 플로팅.

plt.plot(day, weight_pred, 'r-')
plt.scatter(day,weight,label='Day vs Weight')
plt.title('Day vs Weight a*b^x')
plt.xlabel('Day')
plt.ylabel('Weight')
plt.legend()
plt.show()


# In[17]:


# Equation .

a=popt[0].round(4)
b=popt[1].round(4)
print(f'The equation of regression line is y={a}*{b}^x')


# In[18]:


#Check the accuracy
#모델의 정확성을 확인하십시오.
#r2_score로 모델의 정확성을 확인하십시오

from sklearn.metrics import r2_score

r2_score(weight,weight_pred)

