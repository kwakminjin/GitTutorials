#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime as dt
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from  scipy.stats import skew, kurtosis, shapiro


# In[20]:


path = "c:\spain_energy_market.csv"
data = pd.read_csv(path, sep=",", parse_dates=["datetime"])
#parse_dates는 날짜를 datetime 형태로 변환할지 여부인데, True라고 했으니 해당 컬럼에 있는 데이터는 날짜 형태가 된다.
data = data[data["name"]=="Demanda programada PBF total"]#.set_index("datetime")
data["date"] = data["datetime"].dt.date ## YYYY-MM-DD(문자) 반환형태
data.set_index("date", inplace=True) #date열을 인덱스로 사용
data = data[["value"]] #value값만 추출
data = data.asfreq("D") #asfreq는 Datetime Index를 원하는 주기로 나누어주는 메서드 입니다.
#이러한 방식을 리샘플링(resample)이라고 합니다. D = day
data = data.rename(columns={"value": "energy"})
data.info()          


# In[21]:


data[:5]


# In[22]:


data.plot(title="Energy Demand")
plt.ylabel("MWh")
plt.show()


# In[23]:


len(pd.date_range(start="2014-01-01", end="2018-12-31"))


# In[24]:


data["year"] = data.index.year
data["qtr"] = data.index.quarter #분기(숫자)
data["mon"] = data.index.month
data["week"] = data.index.week
data["day"] = data.index.weekday #정수로 요일을 반환합니다. 월요일은 0이고 일요일은 6
data["ix"] = range(0,len(data))
data[["movave_7", "movstd_7"]] = data.energy.rolling(7).agg([np.mean, np.std]) #7일 이동평균값계산 rolling : 이동평균계산
data[["movave_30", "movstd_30"]] = data.energy.rolling(30).agg([np.mean, np.std]) #agg : 다중집계작업
data[["movave_90", "movstd_90"]] = data.energy.rolling(90).agg([np.mean, np.std])
data[["movave_365", "movstd_365"]] = data.energy.rolling(365).agg([np.mean, np.std])

plt.figure(figsize=(20,16))
data[["energy", "movave_7"]].plot(title="Daily Energy Demand in Spain (MWh)")
plt.ylabel("(MWh)")
plt.show()


# In[25]:


mean = np.mean(data.energy.values)
std = np.std(data.energy.values)
skew = skew(data.energy.values) #왜도
ex_kurt = kurtosis(data.energy) #첨도
print("Skewness: {} \nKurtosis: {}".format(skew, ex_kurt+3))


# In[26]:


def shapiro_test(data, alpha=0.05):
    stat, pval = shapiro(data) #검정통계량, pvalue
    print("H0: Data was drawn from a Normal Ditribution")
    if (pval<alpha):
        print("pval {} is lower than significance level: {}, therefore null hypothesis is rejected".format(pval, alpha))
    else:
        print("pval {} is higher than significance level: {}, therefore null hypothesis cannot be rejected".format(pval, alpha))
        
shapiro_test(data.energy, alpha=0.05)


# In[27]:


sns.distplot(data.energy)
plt.title("Target Analysis")
plt.xticks(rotation=45)
plt.xlabel("(MWh)")
plt.axvline(x=mean, color='r', linestyle='-', label="\mu: {0:.2f}%".format(mean))
plt.axvline(x=mean+2*std, color='orange', linestyle='-')
plt.axvline(x=mean-2*std, color='orange', linestyle='-')
plt.show()


# In[28]:


# Insert the rolling quantiles to the monthly returns
data_rolling = data.energy.rolling(window=90)
data['q10'] = data_rolling.quantile(0.1).to_frame("q10")
data['q50'] = data_rolling.quantile(0.5).to_frame("q50")
data['q90'] = data_rolling.quantile(0.9).to_frame("q90")

data[["q10", "q50", "q90"]].plot(title="Volatility Analysis: 90-rolling percentiles")
plt.ylabel("(MWh)")
plt.show()


# In[31]:


data.groupby("qtr")["energy"].std().divide(data.groupby("qtr")["energy"].mean()).plot(kind="bar")
#divide()로 나누어줌
plt.title("Coefficient of Variation (CV) by qtr")
plt.show()


# In[41]:


data.groupby("mon")["energy"].std().divide(data.groupby("mon")["energy"].mean()).plot(kind="bar")
plt.title("Coefficient of Variation (CV) by month")
plt.show()


# In[42]:


data[["movstd_30", "movstd_365"]].plot(title="Heteroscedasticity analysis")
plt.ylabel("(MWh)")
plt.show()


# In[43]:


data[["movave_30", "movave_90"]].plot(title="Seasonal Analysis: Moving Averages")
plt.ylabel("(MWh)")
plt.show()


# In[44]:


sns.boxplot(data=data, x="qtr", y="energy")
plt.title("Seasonality analysis: Distribution over quaters")
plt.ylabel("(MWh)")
plt.show()


# In[45]:


sns.boxplot(data=data, x="day", y="energy")
plt.title("Seasonality analysis: Distribution over weekdays")
plt.ylabel("(MWh)")
plt.show()


# In[46]:


data_mon = data.energy.resample("M").agg(sum).to_frame("energy")
#datetime 객체인 인덱스를 resample('M') 메서드가 월별로 그룹핑합니다.
data_mon["ix"] = range(0, len(data_mon))
data_mon[:5]


# In[47]:


sns.regplot(data=data_mon,x="ix", y="energy")
plt.title("Trend analysis: Regression")
plt.ylabel("(MWh)")
plt.xlabel("")
plt.show()


# In[48]:


sns.boxplot(data=data["2014":"2017"], x="year", y="energy")
plt.title("Trend Analysis: Annual Box-plot Distribution")
plt.ylabel("(MWh)")
plt.show()

