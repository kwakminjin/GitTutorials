#!/usr/bin/env python
# coding: utf-8

# In[22]:


###1.3 다항 회귀:
# Import required libraries:
## 세타 = (xtx)-1(xty)  -1은 역행렬표기
import numpy as np
import matplotlib.pyplot as plt


# In[29]:


# Generate datapoints:
#데이터 포인트 생성
#다항 회귀를 구현하기위한 데이터 세트를 생성 할 것입니다.

x = np.arange(-5,5,0.1)
y_noise = 20 * np.random.normal(size = len(x))
y = 1*(x**3) + 1*(x**2) + 1*x + 3+y_noise
plt.scatter(x,y)


# In[24]:


# Make polynomial data:
# x, x², x³ 벡터 초기화 :
#x의 최대 거듭 제곱을 3으로 취합니다. 따라서 X 행렬은 X, X², X³를 갖게됩니다.

x1 = x
x2 = np.power(x1,2)
x3 = np.power(x1,3)


# In[30]:


# Reshaping data
n=len(x)
x1_new = np.reshape(x1,(n,1)) #2차원으로 (n,1) 배열
x2_new = np.reshape(x2,(n,1))
x3_new = np.reshape(x3,(n,1))

#print(x1_new)
#print(x3_new)


# In[13]:


# First column of matrix X:
#Coefficient of beta-0 = 1
#X 행렬의 열 -1
#주 행렬 X의 첫 번째 열은 beta_0의 계수를 보유하기 때문에 항상 1입니다.
x_bias = np.ones((n,1))


# In[14]:


# Form the complete x matrix:
# 완전한 x 행렬을 만듭니다.
#이 구현의 시작 부분에서 행렬 X를보십시오. 벡터를 추가하여 생성 할 것입니다.
##np.append 는 두개의 1차원 배열을 합칠 수 있습니다.
#출처: https://nevertrustbrutus.tistory.com/410 [FU11M00N]

x_new = np.append(x_bias,x1_new,axis=1)
x_new = np.append(x_new,x2_new,axis=1)
x_new = np.append(x_new,x3_new,axis=1)


# In[31]:


# Finding transpose:
#행렬 전치 :
#세타 값을 단계별로 계산할 것입니다. 먼저 행렬의 전치를 찾아야합니다.

x_new_transpose = np.transpose(x_new)


# In[32]:


# Finding dot product of original and transposed matrix :
#행렬 곱셈 :
#전치를 찾은 후 원래 행렬과 곱해야합니다. 정규 방정식으로 구현할 것이므로 규칙을 따라야합니다.
#dot은 넘파이행렬곱할때 사용

x_new_transpose_dot_x_new = x_new_transpose.dot(x_new)


# In[33]:


# Finding Inverse:
#역행렬 :
#행렬의 역행렬을 찾아서 temp1에 저장합니다 .

temp_1 = np.linalg.inv(x_new_transpose_dot_x_new)# Finding the dot product of transposed x and y :
temp_2 = x_new_transpose.dot(y)


# In[34]:


# Finding coefficients:
#계수 값 :
#계수 값을 찾으려면 temp1과 temp2를 곱해야합니다. 정규 방정식 공식을 참조하십시오.

theta = temp_1.dot(temp_2)
theta


# In[35]:


# Store coefficient values in different variables:
#계수를 변수에 저장합니다.
#이러한 계수 값을 다른 변수에 저장합니다.
beta_0 = theta[0]
beta_1 = theta[1]
beta_2 = theta[2]
beta_3 = theta[3]


# In[37]:


# Plot the polynomial curve:
#곡선으로 데이터 플로팅 :
#회귀 곡선으로 데이터 플로팅.
plt.scatter(x,y)
plt.plot(x,beta_0 + beta_1*x1 + beta_2*x2 + beta_3*x3,c="red")


# In[39]:


# Prediction function:
#예측 기능 :
#이제 회귀 곡선을 사용하여 출력을 예측할 것입니다.
def prediction(x1,x2,x3,beta_0,beta_1,beta_2,beta_3):
    y_pred = beta_0 + beta_1*x1 + beta_2*x2 + beta_3*x3
    return y_pred
# Making predictions:
pred = prediction(x1,x2,x3,beta_0,beta_1,beta_2,beta_3)


# In[41]:


# Calculate accuracy of model:
#오류 기능 :
#평균 제곱 오차 함수를 사용하여 오차를 계산합니다.
def err(y_pred,y):
    var = (y - y_pred)
    var = var*var
    n = len(var)
    MSE = var.sum()
    MSE = MSE/n
    
    return MSE


# In[42]:


# Calculating the error:
#오류를 계산하십시오.

error = err(pred,y)
error

