#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train = pd.read_csv('C:/train.csv')
test = pd.read_csv('C:/test.csv')


# In[3]:


print(train.shape)
print(test.shape)


# In[4]:


train.head()


# In[5]:


sns.heatmap(train.isnull(), yticklabels = False, cbar = False)

#we can average fill the missing fields for age. 
#but cabin seem to have too many missing information, we might need to drop the column.


# In[6]:


train.info()


# In[7]:


train.describe()

#a few columns i'm interested to explore if it affects survivability is:
    #fare 
    #sibsp
    #parch
    #age


# In[8]:


sns.displot(data = train, x = 'Age')


# In[9]:


sns.displot(data = train, x = 'Fare', bins = 10)


# In[ ]:


sns.displot(data = train, x = 'SibSp', bins = 10)


# In[10]:


sns.displot(data = train, x = 'Parch')


# In[11]:


pd.pivot_table(data = train, index = 'Survived', values = ['Age', 'Fare', 'SibSp', 'Parch'])


# In[12]:


#Preliminary look at the data we deduce: 
    #Age is normally distributed. 
    #Bulk of the passengers paid for a cheaper ticket. 
    #Bulk of the passengers don't have any siblings or spouse. 
    #Bulk of the passengers don't have parent or child.


# In[13]:


sns.heatmap(train.isnull(), yticklabels = False, cbar = False)

#fill the missing fields for age with average.
#but cabin seem to have too many missing information, we to drop the column.


# In[15]:


train['Age'].fillna(train['Age'].mean(), inplace = True)


# In[16]:


train.drop('Cabin', axis = 1, inplace = True)


# In[17]:


sns.heatmap(train.isnull(), yticklabels = False, cbar = False)


# In[18]:


train.head()


# In[19]:


#lets see if the title of passengers affect survivability

train['title'] = train.Name.apply(lambda x : x.split(',')[1].split('.')[0].strip())

#strip() 함수 : 문자열 앞뒤의 공백 또는 특별한 문자 삭제


# In[20]:


train.head(1)


# In[24]:


pd.pivot_table(data = train, index = 'Survived', columns = 'title', values = 'Age', aggfunc = 'count')


# In[ ]:


#the captain of the ship did not survive. We can assume that he tried his best to get passengers off before himself.
#col, major, sir - military personnels are more likely to survive.
#doctors are also very likely to survive (3/4)
#the only countess onboard survived. She was later praised for helping row the safety craft to a rescue ship.
#most passengers fall into the Miss, Mr, and Mrs categories. Of which, male title (Mr) is the most likely not to survive

#so yes, title has a correlation to survivability


# In[25]:


#lets see if the class of passengers affect survivability

sns.countplot(data = train, x = 'Survived', hue = 'Pclass')

#3rd class passengers will most likely not survive.


# In[27]:


#lets see if the age of passengers affect survivability

bins = [18, 30, 40, 50, 60, 70, 120]
labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']
train['age_range'] = pd.cut(train.Age.astype(np.int64), bins, labels = labels, include_lowest = True)


# In[28]:


sns.countplot(data = train, x = 'Survived', hue = 'age_range')


# In[29]:


train.age_range.value_counts()


# In[ ]:


#in both categories: survived and did not survive, the majority are aged 18-29.
#in both categories: slowly trickles down in count as the age range increases.
#this is becos most passengers are aged 18-29. Naturally, this would result in both categories having the same distribution.
#no conclusive result can be derived.
#not used.


# In[30]:


train.head()


# In[31]:


train.head(2)


# In[32]:


test.head(2)


# In[33]:


sns.heatmap(test.isnull(), yticklabels = False, cbar = False)


# In[35]:


test.drop('Cabin', axis = 1, inplace = True)


# In[37]:


test['Age'].fillna(train['Age'].mean(), inplace = True)


# In[38]:


test['title'] = test.Name.apply(lambda x : x.split(',')[1].split('.')[0].strip())


# In[ ]:


bins = [18, 30, 40, 50, 60, 70, 120]
labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']
test['age_range'] = pd.cut(test.Age.astype(np.int64), bins, labels = labels, include_lowest = True)


# In[ ]:


test


# In[ ]:


sns.heatmap(test.isnull(), yticklabels = False, cbar = False)


# In[ ]:


#replace the null values with most common age_range - 18 to 29 yo

test.age_range.value_counts()


# In[ ]:


test['age_range'].fillna('18-29', inplace = True)


# In[ ]:


sns.heatmap(test.isnull(), yticklabels = False, cbar = False)


# In[ ]:


#do the same for train dataset

train['age_range'].fillna('18-29', inplace = True)
sns.heatmap(train.isnull(), yticklabels = False, cbar = False)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train_sex = pd.get_dummies(train['Sex'], drop_first = True) #cos if one column alr have 1 and 0 there is no need for another (1 for male)
test_sex = pd.get_dummies(train['Sex'], drop_first = True)


# In[ ]:


train_embarked = pd.get_dummies(train['Embarked'])
test_embarked = pd.get_dummies(test['Embarked'])


# In[ ]:


train_age_range = pd.get_dummies(train['age_range'])
test_age_range = pd.get_dummies(test['age_range'])


# In[ ]:


train_title = pd.get_dummies(train['title'])
test_title = pd.get_dummies(test['title'])


# In[ ]:


train = pd.concat([train,train_sex,train_embarked, train_age_range, train_title], axis = 1)
test = pd.concat([test,test_sex,test_embarked, test_age_range, test_title], axis = 1)


# In[ ]:


train.head(1)


# In[ ]:


#Multicollinearity is instance where some columns are perfect predictors of other columns
#reduces the precision of the estimated coefficients, 
#which weakens the statistical power of your regression model. 
#You might not be able to trust the p-values to identify independent variables that are statistically significant.

#hence need to drop columns with multicollinearity
#in this case - sex


# In[ ]:


train.drop(['Name', 'Sex', 'Embarked', 'Ticket', 'title', 'age_range','Fare'], inplace = True, axis =1)
test.drop(['Name', 'Sex', 'Embarked', 'Ticket', 'title', 'age_range','Fare'], inplace = True, axis =1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train['Dona'] = 0 #adding this here becos it is a title present in test set that's missing from train set


# In[ ]:


train['Dona'] = 0 #adding this here becos it is a title present in test set that's missing from train set


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression(solver='liblinear')


# In[ ]:


logmodel.fit(x_train,y_train)


# In[ ]:


predictions = logmodel.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


np.any(np.isnan(test))


# In[ ]:


np.all(np.isfinite(test))


# In[ ]:


test.shape 

#ran into an issue where i cant use trained model on my test dataset cos my pd.get dummies
#resulted in making nan rows which caused errors


# In[ ]:


train.shape


# In[ ]:


test.dropna(inplace = True)


# In[ ]:


test.shape


# In[ ]:


test.head(1)


# In[ ]:


train.head(1)


# In[ ]:


#ran into another issue where train dataset has more titles than test data set 
#hence model couldnt run due to test set having 24 features and model expecting 32 cos it was trained on 32
#solution to add empty columns of missing titles


# In[ ]:


test.columns


# In[ ]:


train.columns


# In[ ]:


test['Capt'] = 0
test['Don'] = 0
test['Jonkheer'] = 0
test['Lady'] = 0
test['Major'] = 0
test['Mlle'] = 0
test['Mme'] = 0
test['Sir'] = 0
test['the Countess'] = 0


# In[ ]:


test_preds = logmodel.predict(test)


# In[ ]:


test_ids = test["PassengerId"]


# In[ ]:


submission = pd.DataFrame({"PassengerId":test_ids.values,
                  "Survived":test_preds})

submission['PassengerId'] = submission['PassengerId'].astype(int)


# In[ ]:


submission


# In[ ]:


submission.to_csv("submission.csv",index = False)

