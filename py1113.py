#!/usr/bin/env python
# coding: utf-8

# In[222]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# In[236]:


df = pd.read_csv('C:\Complete Steven Wilson.csv')


# In[226]:


print(df.head())


# In[227]:


print(df.shape)


# In[228]:


print(df.info())


# In[229]:


print(df['artist'].value_counts())


# In[230]:


plt.hist(df['loudness'])
plt.xlabel('loudness')
plt.ylabel('Value')
plt.show()


# In[237]:


data_to_boxplot = [df['acousticness'], df['danceability'], df['duration_ms'], df['energy'],
       df['instrumentalness'], df['key'], df['liveness'], df['loudness'], df['speechiness'],
       df['tempo'], df['time_signature'], df['valence']]
plt.boxplot(data_to_boxplot)
plt.xlabel('Attributes')
plt.ylabel('Value')
plt.show()


# In[266]:


print(df.isnull().sum())


# In[ ]:





# In[ ]:





# In[267]:


df.drop(columns=['album', 'analysis_url', 'id', 'mode', 'name', 'track_href',
                'type', 'uri', 'lyrics', 'time_signature'], inplace=True)


# In[234]:


df.columns


# In[ ]:





# In[268]:


training_points = df.drop(columns=['artist'])
training_labels = df['artist']
X_train, X_test, y_train, y_test = train_test_split(
training_points,
training_labels,
test_size=0.3,
random_state=4)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[269]:


classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)
guesses = classifier.predict(X_test)
print(guesses)
print(confusion_matrix(y_test, guesses))
print(metrics.accuracy_score(y_test, guesses))


# In[270]:


k_range = range(1, 50)
accuracy_scores = []
for k in k_range:
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(X_train, y_train)
    guesses = classifier.predict(X_test)
    accuracy_scores.append(metrics.accuracy_score(y_test, guesses))
print(accuracy_scores)

#Visualize the result of KNN accuracy with matplotlib
plt.plot(k_range, accuracy_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()


# In[271]:


classifier = KNeighborsClassifier(n_neighbors = 4)
classifier.fit(X_train, y_train)
guesses = classifier.predict(X_test)
print(guesses)
print(confusion_matrix(y_test, guesses))
print(metrics.accuracy_score(y_test, guesses))


# In[182]:


df.columns


# In[246]:


from sklearn.preprocessing import StandardScaler

#Create copy of dataset.
df_model = df.copy()

scaler = StandardScaler()

features = [['acousticness', 'danceability', 'duration_ms', 'energy',
            'instrumentalness', 'key', 'liveness', 'loudness', 'speechiness',
       'tempo', 'valence']]

for feature in features:
    df_model[feature] = scaler.fit_transform(df_model[feature])

training_points = df_model.drop(columns=['artist'])
training_labels = df_model['artist']


# In[247]:


X_train, X_test, y_train, y_test = train_test_split(
training_points,
training_labels,
test_size=0.3,
random_state=4)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[248]:


k_range = range(1, 50)
accuracy_scores = []
for k in k_range:
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(X_train, y_train)
    guesses = classifier.predict(X_test)
    accuracy_scores.append(metrics.accuracy_score(y_test, guesses))
print(accuracy_scores)

#Visualize the result of KNN accuracy with matplotlib
plt.plot(k_range, accuracy_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()


# In[251]:


classifier = KNeighborsClassifier(n_neighbors = 24)
classifier.fit(X_train, y_train)
guesses = classifier.predict(X_test)
print(guesses)
print(confusion_matrix(y_test, guesses))
print(metrics.accuracy_score(y_test, guesses))


# In[252]:


from sklearn.preprocessing import MinMaxScaler

#Create copy of dataset.
df_model = df.copy()

scaler = MinMaxScaler()

features = [['acousticness', 'danceability', 'duration_ms', 'energy',
            'instrumentalness', 'key', 'liveness', 'loudness', 'speechiness',
       'tempo', 'valence']]

for feature in features:
    df_model[feature] = scaler.fit_transform(df_model[feature])

training_points = df_model.drop(columns=['artist'])
training_labels = df_model['artist']


# In[253]:


X_train, X_test, y_train, y_test = train_test_split(
training_points,
training_labels,
test_size=0.3,
random_state=4)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[254]:


k_range = range(1, 50)
accuracy_scores = []
for k in k_range:
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(X_train, y_train)
    guesses = classifier.predict(X_test)
    accuracy_scores.append(metrics.accuracy_score(y_test, guesses))
print(accuracy_scores)

#Visualize the result of KNN accuracy with matplotlib
plt.plot(k_range, accuracy_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()


# In[256]:


classifier = KNeighborsClassifier(n_neighbors = 29)
classifier.fit(X_train, y_train)
guesses = classifier.predict(X_test)
print(guesses)
print(confusion_matrix(y_test, guesses))
print(metrics.accuracy_score(y_test, guesses))


# In[257]:


from sklearn.preprocessing import RobustScaler
#Create copy of dataset.
df_model = df.copy()

#Rescaling features age, trestbps, chol, thalach, oldpeak.
scaler = RobustScaler()

features = [['acousticness', 'danceability', 'duration_ms', 'energy',
            'instrumentalness', 'key', 'liveness', 'loudness', 'speechiness',
       'tempo', 'valence']]

for feature in features:
    df_model[feature] = scaler.fit_transform(df_model[feature])

training_points = df_model.drop(columns=['artist'])
training_labels = df_model['artist']


# In[258]:


X_train, X_test, y_train, y_test = train_test_split(
training_points,
training_labels,
test_size=0.3,
random_state=4)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[259]:


k_range = range(1, 50)
accuracy_scores = []
for k in k_range:
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(X_train, y_train)
    guesses = classifier.predict(X_test)
    accuracy_scores.append(metrics.accuracy_score(y_test, guesses))
print(accuracy_scores)

#Visualize the result of KNN accuracy with matplotlib
plt.plot(k_range, accuracy_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()


# In[265]:


classifier = KNeighborsClassifier(n_neighbors = 27)
classifier.fit(X_train, y_train)
guesses = classifier.predict(X_test)
print(guesses)
print(confusion_matrix(y_test, guesses))
print(metrics.accuracy_score(y_test, guesses))


# In[ ]:





# In[ ]:





# In[ ]:




