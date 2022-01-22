#!/usr/bin/env python
# coding: utf-8

# In[90]:


# data analysis and wrangling #데이터 분석 및 논쟁
import pandas as pd
import numpy as np
import random as rnd

# visualization #시각화
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning #머신러닝
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[91]:


train_df = pd.read_csv('c:/train.csv')
test_df = pd.read_csv('c:/test.csv')
combine = [train_df, test_df]


# In[92]:


print(train_df.columns.values)


# In[93]:


train_df.head()


# In[94]:


train_df.info()
print('_'*40)
test_df.info()


# In[95]:


train_df.describe()
# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
# Review Parch distribution using `percentiles=[.75, .8]`
# SibSp distribution `[.68, .69]`
# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`


# In[96]:


train_df.describe(include=['O']) #문자열(String) select_dtypes (ex: df.describe (include=['0']))


# In[97]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'],
                                         as_index=False).mean().sort_values(by='Survived',
                                                                            ascending=False)
#as_index=False 구문은 이 그룹을 인덱스로 지정할 것인지 여부(지정하면 pclass가 인덱스가됨)


# In[98]:


train_df[["Sex", "Survived"]].groupby(['Sex'],
                                      as_index=False).mean().sort_values(by='Survived',
                                                                         ascending=False)


# In[99]:


train_df[["SibSp", "Survived"]].groupby(['SibSp'],
                                        as_index=False).mean().sort_values(by='Survived',
                                                                           ascending=False)


# In[100]:


train_df[["Parch", "Survived"]].groupby(['Parch'],
                                        as_index=False).mean().sort_values(by='Survived',
                                                                           ascending=False)


# In[101]:


g = sns.FacetGrid(train_df, col='Survived') #패싯그리드
#다양한 범주형 값을 가지는 데이터를 시각화하는데 좋은 방법
#행, 열 방향으로 서로 다른 조건을 적용하여 여러 개의 서브 플롯 제작
#각 서브 플롯에 적용할 그래프 종류를 map() 메서드를 이용하여 그리드 객체에 전달

#1. FacetGrid에 데이터프레임과 구분할 row, col, hue 등을 전달해 객체 생성

#2. 객체(facet)의 map 메서드에 그릴 그래프의 종류와 종류에 맞는 컬럼 전달
#예시 - distplot의 경우 하나의 컬럼 // scatter의 경우 두개의 컬럼

g.map(plt.hist, 'Age', bins=20) #x값 age


# In[102]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
#컬럼엔 생존자수를, 로우엔 Pclass를 넣었다.
#두 컬럼 집단별 나이의 분포를 볼 수 있게 됐다.

 
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# In[103]:


# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# In[104]:


# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# In[105]:


print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
#shape 하면 해당 차원이 몇 차원인지 표시해줍니다.

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape


# In[106]:


print(combine)


# In[56]:


print(train_df)


# In[107]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
#알파벳(대,소)중하나의문자 1개이상에다가 .(리터럴점)
#'Title' 열을 새로 만들고, dataset에서 Name열의 str(문자열) 중에서 ' ([A-Za-z]+)\.'를 extract(추출)하여 'Title' 열에 넣겠다는 의미
# expand=True이면 여러 컬럼, False이면 1개 컬럼에 리스트

pd.crosstab(train_df['Title'], train_df['Sex']) #왜 train_df에도 Title열이 생기는지는 모르겠음
#교차표

#print(train_df)


# In[58]:


print(train_df)


# In[108]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[109]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping) #'Title'열을 맵핑시킴
    dataset['Title'] = dataset['Title'].fillna(0) #모든 NaN값을 0으로 바꿈

train_df.head()


# In[110]:


train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)

#① aixs=0(index)은 행을 따라 동작합니다. 각 컬럼의 모든 행에 대해서 작용합니다. ② aixs=1(columns)은 열을 따라 동작합니다. 각 행의 모든 컬럼에 대해서 작동합니다.

combine = [train_df, test_df]
train_df.shape, test_df.shape


# In[111]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()


# In[112]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# In[113]:


guess_ages = np.zeros((2,3)) #0으로 가득찬 2 * 3 배열만들기
guess_ages


# In[114]:


#성별(0 또는 1)과 P클래스(1, 2, 3)에 대해 반복하여 6가지 조합에 대한 연령의 추측 값을 계산한다.

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()
            
            # 위에서 guess_ages사이즈를 [2,3]으로 잡아뒀으므로 j의 범위도 이를 따름

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            # age의 random값의 소수점을 .5에 가깝도록 변형
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]
# loc을 이용해서 데이터에 접근하기

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()


# In[115]:


#나이대(age band) 특성을 만들고 생존율과 상관 관계를 추측한다.

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[118]:


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()


# In[119]:


#age값을 수정했으니 age band 특성은 이제 제거해준다.

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()


# In[120]:


#우리는 Parch 특성과 SibSp 특성을 합쳐 FamilySize 특성을 만들 수 있다.
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[121]:


#가족이 없는 사람은 따로 가중치를 적용할 수 있도록 isAlone 이라는 특성을 만든다.
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# In[122]:


#다시 Parch, SibSp 특성을 제거한다.
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()


# In[123]:


#이제 pclass 와 나이를 합친 혼합 특성을 만들 수 있다.
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# In[124]:


#Embarked 특성은 S, Q, C 의 데이터를 가진 범주형 특성이다. 그리고 Embarked 특성은 빈 데이터가 상당히 많은데, 가장 많이 발견되는 데이터인 S로 채운다.

freq_port = train_df.Embarked.dropna().mode()[0] #최빈값 반환
freq_port


# In[125]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[126]:


#Embarked 특성이 이제 빈 자리가 없으니 다시 숫자형 데이터로 바꿔주자.

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()


# In[127]:


#우리는 이제 누락이 많은 Fare 퀄럼 데이터에 대해서 값을 채워넣을 수 있다.
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()


# In[128]:


#이제 이 Fare 데이터를 다시 범위형 데이터로 변경한다.
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[129]:


#그리고 그 범위형 데이터를 순서형 데이터로 다시 변경하고 이제 범위형 데이터 컬럼을 제거한다.
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)


# In[130]:


test_df.head(10)


# In[133]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[134]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[136]:


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature'] #열이름
coeff_df["Correlation"] = pd.Series(logreg.coef_[0]) #시리즈 생성

coeff_df.sort_values(by='Correlation', ascending=False)


# In[138]:


# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[139]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[142]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[143]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[144]:


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# In[145]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[146]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[147]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[148]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[149]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
# submission.to_csv('../output/submission.csv', index=False)

