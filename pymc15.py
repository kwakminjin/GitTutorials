#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import numpy as np
import pandas as pd
customers = pd.read_csv("c:\FyntraCustomerData.csv")


# In[3]:


customers.head()


# In[4]:


customers.describe()


# In[5]:


customers.info()


# In[6]:


#Check Correlations
# More time on site, more money spent.
sns.jointplot(x='Time_on_Website',y='Yearly_Amount_Spent',data=customers)


# In[7]:


correlation = customers.corr()


# In[8]:


sns.heatmap(correlation, cmap="YlGnBu")


# In[9]:


sns.jointplot(x='Time_on_App',y='Yearly_Amount_Spent',data=customers)
# This one looks stronger correlation than Time_on_Website


# In[10]:


sns.pairplot(customers)


# In[11]:


sns.lmplot(x='Length_of_Membership',y='Yearly_Amount_Spent',data=customers)


# In[12]:


sns.jointplot(x='Length_of_Membership', y='Yearly_Amount_Spent', data=customers,kind="kde")
#


# In[13]:


X = customers[['Avg_Session_Length', 'Time_on_App','Time_on_Website', 'Length_of_Membership']]


# In[14]:


y = customers['Yearly_Amount_Spent']


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=85)


# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


lm = LinearRegression()


# In[19]:


lm.fit(X_train,y_train)


# In[20]:


#calculating the residuals
print('y-intercept             :' , lm.intercept_)
print('beta coefficients       :' , lm.coef_)


# In[21]:


predictions = lm.predict( X_test)


# In[22]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test ')
plt.ylabel('Y Predicted ')


# In[23]:


# here You can check the values Test vrs Prediction
dft = pd.DataFrame({'Y test': y_test, 'Y Pred':predictions})
dft.head(10)


# In[24]:


# calculate these metrics by hand!
from sklearn import metrics

print('Mean Abs Error MAE      :' ,metrics.mean_absolute_error(y_test,predictions))
print('Mean Sqrt Error MSE     :' ,metrics.mean_squared_error(y_test,predictions))
print('Root Mean Sqrt Error RMSE:' ,np.sqrt(metrics.mean_squared_error(y_test,predictions)))
print('r2 value                :' ,metrics.r2_score(y_test,predictions))


# In[35]:


sns.distplot((y_test-predictions),bins=50); #-predictors는 무엇을 의미?


# In[30]:


coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients

