#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[37]:


df=pd.read_csv("c:/Life Expectancy Data.csv")


# In[38]:


df.head()


# In[39]:


df.dtypes


# In[40]:


#Analyzing the data
df.describe()


# In[41]:


#Renaming some columns
df.rename({'Life expectancy ':'Life expectancy',' BMI ':'BMI',' HIV/AIDS':'HIV/AIDS',' thinness  1-19 years':'thinness  1-19 years',' thinness 5-9 years':'thinness 5-9 years'},axis=1,inplace=True)
df.rename({'Measles ':'Measles','under-five deaths ':'under-five deaths','Diphtheria ':'Diphtheria',})


# In[42]:


#FINDING RELATION OF NULL VALUE 


# In[43]:


list_of_null_values=dict(df.isnull().sum())


# In[44]:


null_value_cols=[]
for key,value in list_of_null_values.items():
    if value>0:
        null_value_cols.append(key)


# In[45]:


null_value_cols=null_value_cols[1:]


# In[46]:


plt.figure(figsize=(20,20))

for i in range(len(null_value_cols)):
    if i<12:
        plt.subplot(5,3,i+1)
        plt.scatter(df[null_value_cols[i]],df['Life expectancy'])
        plt.xlabel(null_value_cols[i])
        plt.ylabel('Life Expectancy')
    else:
        plt.subplot(5,3,i+1)
        plt.scatter(df[null_value_cols[i]],df['Life expectancy'])
        plt.xlabel(null_value_cols[i])
        plt.ylabel('Life Expectancy')
        
    

plt.tight_layout()


# In[47]:


#Identifying relationship between Country Status and Life Expectancy 
plt.figure(figsize=(10,10))
df.groupby(['Status'])['Life expectancy'].mean().plot(kind='bar')


# In[50]:


df.columns


# In[51]:


plt.figure(figsize=(10,10))
plt.scatter(df['Schooling'],df['Life expectancy'],c='pink')
plt.xlabel('Schooling')
plt.ylabel('Life Expectancy')


# In[52]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True)


# In[53]:


plt.figure(figsize=(50,15))
val=df.groupby('Country')['Life expectancy'].mean().sort_values(ascending=False).plot(kind='bar',fontsize=15)


# In[54]:


plt.figure(figsize=(50,15))
val=df.groupby('Country')['Adult Mortality'].mean().sort_values(ascending=False).plot(kind='bar',fontsize=15)


# In[55]:


plt.figure(figsize=(50,15))
val=df.groupby('Country')['infant deaths'].mean().sort_values(ascending=False).plot(kind='bar',fontsize=15)


# In[56]:


plt.figure(figsize=(50,15))
val=df.groupby('Country')['GDP'].mean().sort_values(ascending=False).plot(kind='bar',fontsize=15)


# In[57]:


plt.figure(figsize=(50,15))
val=df.groupby('Schooling')['GDP'].mean().sort_values(ascending=False).plot(kind='bar',fontsize=15)


# In[58]:


#HANDLING NULL VALUES
data=df.copy()


# In[59]:


data.isnull().sum()


# In[60]:


list_of_null_values=dict(data.isnull().sum())


# In[61]:


data.isnull().sum()


# In[62]:


null_fields=[]
list_of_null_values=dict(data.isnull().sum())
for key,value in list_of_null_values.items():
    if value>0:
        null_fields.append(key)


# In[63]:


empty_list=[]
for year in list(df.Year.unique()):
    year_data = df[df.Year == year].copy()
    for col in list(year_data.columns)[3:]:
        year_data[col] = year_data[col].fillna(year_data[col].dropna().mean()).copy()
    empty_list.append(year_data)
data = pd.concat(empty_list).copy()
#데이터프레임을 위/아래(행 기준)로 합치거나 옆으로(열 기준)으로 합치기


# In[69]:


data.reset_index(inplace=True)
#기존에 설정되어 있는 행 인덱스를 제거하고 그 인덱스를 데이터 열(columns)로 추가하는 방식
data.drop('index',axis=1,inplace=True)


# In[71]:


data['Life_expectancy']=data['Life expectancy']


# In[73]:


data.drop('Life expectancy',axis=1,inplace=True)


# In[72]:


data.head()


# In[80]:


plt.figure(figsize=(20,20))
column_list=list(data.columns)[3:]
plt_num=1
for i in column_list:
    if plt_num<=18:
        plt.subplot(5,4,plt_num)
        #행, 열, 몇번째에 그릴지
        #여러 개의 그래프를 하나의 그림에 나타내도록
        sns.histplot(data[i])
        plt_num=plt_num+1
    else:
        plt.subplot(5,4,plt_num)
        sns.histplot(data[i])
        plt_num=plt_num+1
plt.tight_layout()


# In[75]:


for i in column_list:
    iqr=data[i].quantile(0.75)-data[i].quantile(0.25)
    lower_boundary=data[i].quantile(0.25)-1.5*iqr
    upper_boundary=data[i].quantile(0.75)+1.5*iqr
    print("The column {} has {} outliers percentage {} %".format(i,data[data[i]> upper_boundary].shape[0]+data[data[i]< lower_boundary].shape[0],(data[data[i]> upper_boundary].shape[0]+data[data[i]< lower_boundary].shape[0])*100/2938))


# In[79]:


for i in column_list:
    IQR=data[i].quantile(0.75)-data[i].quantile(0.25)
    lower_boundary=data[i].quantile(0.25)-(IQR*1.5)
    upper_boundary=data[i].quantile(0.75)+(IQR*1.5)
    data.loc[data[i]>upper_boundary,i]=upper_boundary
    #upper이상은 upper로
    data.loc[data[i]<lower_boundary,i]=lower_boundary


# In[81]:


plt.figure(figsize=(20,20))
column_list=list(data.columns)[3:]
plt_num=1
for i in column_list:
    if plt_num<=18:
        plt.subplot(5,4,plt_num)
        sns.histplot(data[i])
        plt_num=plt_num+1
    else:
        plt.subplot(5,4,plt_num)
        sns.histplot(data[i])
        plt_num=plt_num+1
plt.tight_layout()


# In[82]:


data['Status']=data['Status'].map({'Developing':0,'Developed':1})


# In[83]:


data=data.iloc[:,2:] #인덱스로 접근


# In[84]:


data


# In[85]:


data_copy1=data.iloc[:,1:]
data_copy2=data['Status']


# In[87]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
data_scaled=scaler.fit_transform(data_copy1.iloc[:,:])


# In[88]:


modified_data=pd.DataFrame(data_scaled,columns=data_copy1.columns)


# In[89]:


new_data=pd.concat([data_copy2,modified_data],axis=1)
#데이터프레임을 위/아래(행 기준)로 합치거나 옆으로(열 기준)으로 합치기


# In[ ]:


#Splitting Data into train and test and applying machine learning models


# In[90]:


X=new_data.iloc[:,:-1]
y=new_data.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[91]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression


# In[92]:


select=SelectKBest(score_func=f_regression,k=12)
z=select.fit(X_train,y_train)
imp_columns=X_train.columns[z.get_support()]


# In[93]:


X_train=X_train[list(imp_columns)]


# In[94]:


from sklearn.linear_model import LinearRegression
lr_model=LinearRegression()


# In[95]:


lr_model.fit(X_train,y_train)


# In[96]:


X_test=X_test[list(imp_columns)]
predictions=lr_model.predict(X_test)


# In[97]:


from sklearn.metrics import r2_score


# In[98]:


ml_model_accuracy={}
ml_model_accuracy['Regression_model']=r2_score(y_test,predictions)


# In[99]:


#knn method
from sklearn.neighbors import KNeighborsRegressor


# In[100]:


empty_list=[]
for i in range(1,51):
    knn_model=KNeighborsRegressor(n_neighbors=i)
    knn_model.fit(X_train,y_train)
    preds=knn_model.predict(X_test)
    empty_list.append(r2_score(y_test,preds))


# In[101]:


plt.figure(figsize=(5,5))
plt.plot(range(1,51),empty_list)


# In[102]:


knn_model=KNeighborsRegressor(n_neighbors=27)
knn_model.fit(X_train,y_train)
preds=knn_model.predict(X_test)
ml_model_accuracy['knn model']=r2_score(y_test,preds)


# In[103]:


#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rfr_model=RandomForestRegressor()


# In[104]:


rfr_model.fit(X_train,y_train)


# In[105]:


preds=rfr_model.predict(X_test)


# In[106]:


ml_model_accuracy['Random forest Regressor']=r2_score(y_test,preds)


# In[107]:


plt.figure(figsize=(10,10))
keys=list(ml_model_accuracy.keys())
values=list(ml_model_accuracy.values())
plt.bar(range(len(ml_model_accuracy)),values,tick_label=keys)

