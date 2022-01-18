#!/usr/bin/env python
# coding: utf-8

# In[3]:


##1.6 로그 회귀:
#대수 성장의 실제 사례:
#지진의 규모.
#소리의 강도.
#용액의 산도.
#용액의 pH 수준.
#화학 반응의 수율.
#상품 생산.
#유아의 성장.
#COVID-19 그래프.

#Y = a + b * ln(X)
#whrere, Y = output, X = input feature, a = the line/curve always passes through (1,a), b = controls the rate of growth or decay

# Import required libraries:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# In[12]:


#데이터 세트 생성 :
# Dataset:
# Y = a + b*ln(X)
X = np.arange(1,50,0.5)
Y = 10 + 2*np.log(X)

#Adding some noise to calculate error!
Y_noise = np.random.rand(len(Y)) #주어진 형태의 난수 어레이를 생성
#print(Y_noise)
Y = Y +Y_noise
plt.scatter(X,Y)


# In[26]:


#행렬 X의 첫 번째 열 :
#여기서 우리는 계수 값을 찾기 위해 정규 방정식을 사용할 것입니다.

# 1st column of our X matrix should be 1:
n = len(X)
x_bias = np.ones((n,1)) #1로 가득찬 n * 1 array를 생성합니다.
print (X.shape) #NumPy의 배열의 형태 shape() 98 1차원배열???
print (x_bias.shape) #98 * 1의 2차원 배열


# In[14]:


# Reshaping X :
#X 모양 변경 :
X = np.reshape(X,(n,1))
print (X.shape)


# In[15]:


#정규 방정식 공식으로 이동 :
# Going with the formula:
# Y = a + b*ln(X)
X_log = np.log(X)


# In[17]:


#메인 매트릭스 X 형성 :
# Append the X_log to X_bias:
x_new = np.append(x_bias,X_log,axis=1)


# In[18]:


#전치 행렬 찾기 :
# Transpose of a matrix:
x_new_transpose = np.transpose(x_new)


# In[19]:


#행렬 곱셈 수행 :
# Matrix multiplication:
x_new_transpose_dot_x_new = x_new_transpose.dot(x_new)


# In[20]:


#역 찾기 :
# Find inverse:
temp_1 = np.linalg.inv(x_new_transpose_dot_x_new)


# In[21]:


#행렬 곱셈 :
# Matrix Multiplication:
temp_2 = x_new_transpose.dot(Y)


# In[22]:


#계수 값 찾기 :
# Find the coefficient values:
theta = temp_1.dot(temp_2)


# In[24]:


#회귀 곡선을 사용하여 데이터를 플로팅합니다.
# Plot the data:
a = theta[0]
b = theta[1]
Y_plot = a + b*np.log(X)
plt.scatter(X,Y)
plt.plot(X,Y_plot,c="r")


# In[25]:


#정확성체크
# Check the accuracy:
Accuracy = r2_score(Y,Y_plot)
print (Accuracy)

