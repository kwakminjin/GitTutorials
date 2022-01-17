#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1.5 정형파 회귀:
#정현파 회귀의 실제 예 :

#음악 파의 생성.
#소리는 파도로 이동합니다.
#구조물의 삼각 함수.
#우주 비행에 사용됩니다.
#GPS 위치 계산.
#건축물.
#전류.
#라디오 방송.
#바다의 썰물과 만조.
#건물.

# Y = A * sin ( B ( X + C ) + D ) : 정현파 회귀 공식

# Import required libraries:
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


# In[8]:


#데이터세트생성
# Generating dataset:
# Y = A*sin(B(X + C)) + D
# A = Amplitude
# Period = 2*pi/B
# Period = Length of One Cycle
# C = Phase Shift (In Radian)
# D = Vertical Shift
X = np.linspace(0,1,100) #(Start,End,Points)

# Here…
# A = 1
# B= 2*pi
# B = 2*pi/Period
# Period = 1
# C = 0
# D = 0
Y = 1*np.sin(2*np.pi*X)

# Adding some Noise :
Noise = 0.4*np.random.normal(size=100)

Y_data = Y + Noise

plt.scatter(X,Y_data,c="r")

#print(X)
#print(Y_data)


# In[12]:


# Calculate the value:
#사인 함수 적용 :
#여기에서는 최적 계수를 기반으로 출력 값을 계산하기 위해 "calc_sine"이라는 함수를 만들었습니다. 여기에서는 scikit-learn 라이브러리를 사용하여 최적의 매개 변수를 찾습니다.

def calc_sine(x,a,b,c,d):
    return a * np.sin(b* ( x + np.radians(c))) + d

# Finding optimal parameters :
popt,pcov = curve_fit(calc_sine,X,Y_data)
#print(popt)
#popt는  최적의 매개변수

# Plot the main data :
plt.scatter(X,Y_data)# Plot the best fit curve :
plt.plot(X,calc_sine(X,*popt),c="r")

# Check the accuracy :
Accuracy =r2_score(Y_data,calc_sine(X,*popt))
print (Accuracy)


# In[14]:


#정현파 회귀가 선형 회귀보다 나은 이유는 무엇입니까?
#데이터를 직선으로 피팅 한 후 모델의 정확성을 확인하면 예측의 정확성이 사인파 회귀보다 낮다는 것을 알 수 있습니다. 이것이 우리가 정현파 회귀를 사용하는 이유입니다.

# Function to calculate the value :
def calc_line(X,m,b):
    return b + X*m

# It returns optimized parametes for our function :
# popt stores optimal parameters
# pcov stores the covarience between each parameters.
popt,pcov = curve_fit(calc_line,X,Y_data)

# Plot the main data :
plt.scatter(X,Y_data)# Plot the best fit line :
plt.plot(X,calc_line(X,*popt),c="r")

# Check the accuracy of model :
Accuracy =r2_score(Y_data,calc_line(X,*popt))
print ("Accuracy of Linear Model : ",Accuracy)

