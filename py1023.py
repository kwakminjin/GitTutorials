#!/usr/bin/env python
# coding: utf-8

# In[153]:


import pandas as pd

accident_df = pd.read_csv("C:/도로교통공단_사망 교통사고 정보_20201231.csv", encoding = 'cp949')

print(accident_df)


# In[154]:


accident_df.head()


# In[155]:


accident_df.tail()


# In[156]:


accident_df.info()


# In[157]:


accident_df.describe()


# In[158]:


accident_df.columns


# In[159]:


accident_df.drop(columns=['발생년', '중상자수', '경상자수', '부상신고자수', '사고유형_대분류', '사고유형_중분류', '가해자법규위반', '도로형태_대분류', '가해자_당사자종별', '피해자_당사자종별', '발생위치X_UTMK', '발생위치Y_UTMK'], inplace=True)
accident_df.head(5)


# In[160]:


accident_df.columns


# In[161]:


accident_df.rename(columns={'발생년월일시':'발생시각'
}, inplace=True)

accident_df.columns


# In[162]:


accident_df.head(10)


# In[163]:


accident_df.replace({'주야': {'주': 0}}, inplace = True)
accident_df.replace({'주야': {'야': 1}}, inplace = True)
accident_df.head(10)


# In[195]:


accident_df.replace({'요일': {'월': 0}}, inplace = True)
accident_df.replace({'요일': {'화': 1}}, inplace = True)
accident_df.replace({'요일': {'수': 2}}, inplace = True)
accident_df.replace({'요일': {'목': 3}}, inplace = True)
accident_df.replace({'요일': {'금': 4}}, inplace = True)
accident_df.replace({'요일': {'토': 5}}, inplace = True)
accident_df.replace({'요일': {'일': 6}}, inplace = True)

accident_df.head(10)


# In[164]:


accident_df.describe()


# In[165]:


import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic') # 폰트 지정
plt.rc('axes', unicode_minus=False) # 마이너스 폰트 설정

get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina' # 그래프 글씨 뚜렷")


# In[166]:


import seaborn as sns

sns.boxplot(y = accident_df['경도'])


# In[167]:


accident_df.drop(accident_df[accident_df.경도 >= 130].index,
inplace = True)
accident_df.drop(accident_df[accident_df.경도 < 125].index,
inplace = True)
                  
sns.boxplot(y = accident_df['경도'] );


# In[168]:


accident_df.describe()


# In[169]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

accident_df[['경도']] = scaler.fit_transform(accident_df[['경도']])
accident_df[['위도']] = scaler.fit_transform(accident_df[['위도']])
#accident_df[['사망자수']] = scaler.fit_transform(accident_df[['사망자수']])
#accident_df[['부상자수']] = scaler.fit_transform(accident_df[['부상자수']])

accident_df.describe()


# In[170]:


accident_df


# In[172]:


import plotly.graph_objects as go
import pandas as pd

x_values = accident_df['부상자수']

mydata = go.Histogram(x=x_values)

mylayout = go.Layout(title='부상자수 빈도',
xaxis_title = '부상자수',    
yaxis_title="빈도")

fig = go.Figure(data = mydata, layout = mylayout)

fig.show()


# In[175]:


import plotly.graph_objects as go
import pandas as pd

x_values = accident_df['경도']

mydata = go.Histogram(x=x_values)

mylayout = go.Layout(title='경도 빈도',
xaxis_title = '경도',    
yaxis_title="빈도")

fig = go.Figure(data = mydata, layout = mylayout)

fig.show()


# In[174]:


import plotly.graph_objects as go

import numpy as np

x = accident_df['경도']
y = accident_df['위도']

fig = go.Figure(go.Histogram2d(
x=x,
y=y
))
fig.show()


# In[178]:


import plotly.graph_objects as go
import pandas as pd

y_values = accident_df['경도']

mydata = go.Box(y = y_values)

mylayout = go.Layout(title='경도 상자그림', yaxis_title="경도")

fig = go.Figure(mydata, mylayout)

fig.show()


# In[179]:


import plotly.graph_objects as go
import pandas as pd

y_values = accident_df['위도']

mydata = go.Box(y = y_values)

mylayout = go.Layout(title='위도 상자그림', yaxis_title="위도")

fig = go.Figure(mydata, mylayout)

fig.show()


# In[193]:


import plotly.express as px

fig = px.pie(accident_df,
             values='사망자수',
             names='발생지시도')

fig.show()


# In[202]:


fig = px.sunburst(accident_df,
    path=['발생지시도', '발생지시군구'],
    values='사망자수',
    color = '부상자수')

fig.show()


# In[203]:


fig = px.treemap(accident_df,
path=['발생지시도', '발생지시군구'],
values='사망자수',
color='부상자수')
fig.show()


# In[201]:


fig = px.density_contour(accident_df,
x="경도",
y="위도",
color ="발생지시도")
fig.show()


# In[205]:


fig = px.density_heatmap(accident_df,
x="경도",
y="위도")
fig.show()


# In[206]:


sns.catplot(x = "주야", hue="사망자수",
           kind="count", data = accident_df)


# In[207]:


sns.catplot(x = "주야", hue="부상자수",
           kind="count", data = accident_df)


# In[208]:


sns.catplot(x = "요일", hue="사망자수",
           kind="count", data = accident_df)


# In[209]:


sns.catplot(x = "요일", hue="부상자수",
           kind="count", data = accident_df)


# In[ ]:





# In[ ]:





# In[ ]:




