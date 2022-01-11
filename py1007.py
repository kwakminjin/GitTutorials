#!/usr/bin/env python
# coding: utf-8

# In[2]:


import plotly
print(plotly.__version__)


# In[6]:


import numpy
print(numpy.__version__)


# In[3]:


import plotly.graph_objects as go

fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
fig.show()


# In[4]:


import plotly.graph_objects as go

langs = ['C', 'C++', 'Java', 'Python', 'PHP']
students = [23, 17, 35, 29, 12]

data = [go.Bar(
x = langs,
y = students)]

fig = go.Figure(data=data)

fig.show()


# In[9]:


import plotly.graph_objects as go

branches = ['CES', 'Mech', 'Eletronics']

fy = [23, 17, 35]
sy = [20, 23, 30]
ty = [30, 20, 15]


# In[10]:


trace1 = go.Bar(
x = branches,
y = fy,
name = 'FY')

trace2 = go.Bar(
x = branches,
y = sy,
name = 'SY')

trace3 = go.Bar(
x = branches,
y = ty,
name = 'TY')


# In[11]:


data = [trace1, trace2, trace3]
layout = go.Layout(barmode = 'group', title='Departments')
fig = go.Figure(data = data, layout = layout)
fig.show()


# In[12]:


data = [trace1, trace2, trace3]
layout = go.Layout(barmode = 'stack', title='Departments')
fig = go.Figure(data = data, layout = layout)
fig.show()


# In[13]:


import plotly.graph_objs as go

years = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012]

rest = [219, 146, 112, 127, 124, 180, 236, 207, 236, 263, 350, 430,
474, 526, 488, 537, 500, 439]

china = [16, 13, 10, 11, 28, 37, 43, 55, 56, 88, 105, 156, 270, 299,
340, 403, 549, 499]


# In[22]:


trace1 = go.Bar(
x = years,
y = rest,
name = 'Rest of the World',
marker = dict(
color = 'rgb(49,130,189)',
opacity = 0.7
)
)

trace2 = go.Bar(
x = years,
y = china,
name = 'China',
marker = dict(
color = 'rgb(204,204,204)',
opacity = 0.5
)
)


# In[23]:


data = [trace1, trace2]
layout = go.Layout(barmode = 'group', title='Export of Plastic Scrap')
fig = go.Figure(data = data, layout = layout)
fig.show()


# In[24]:


import plotly.graph_objs as go

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
'Oct', 'Nov', 'Dec']

primary_sales = [20, 14, 25, 16, 18, 22, 19, 15, 12, 16, 14, 17]

secondary_sales = [19, 14, 22, 14, 16, 19, 15, 14, 10, 12, 12, 16]


# In[25]:


trace1 = go.Bar(
x = months,
y = primary_sales,
name = 'Primary Product',
marker = dict(
color = 'rgb(49,130,189)',
opacity = 0.7
),
text = primary_sales,
textposition = 'auto',
)


# In[26]:


trace2 = go.Bar(
x = months,
y = secondary_sales,
name = 'Secondary Product',
marker = dict(
color = 'rgb(204,204,204)',
opacity = 0.5
),
text = secondary_sales,
textposition = 'auto',
)


# In[27]:


data = [trace1, trace2]
layout = go.Layout(barmode = 'group', title='2020 Sales Report')
fig = go.Figure(data = data, layout = layout)
fig.show()


# In[61]:


import plotly.graph_objs as go

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
'Oct', 'Nov', 'Dec']

primary_sales = [20, 14, 25, 16, 18, 22, 19, 15, 12, 16, 14, 17]
secondary_sales = [19, 14, 22, 14, 16, 19, 15, 14, 10, 12, 12, 16]


# In[62]:


trace1 = go.Bar(
x = months,
y = primary_sales,
name = 'Primary Product',
marker = dict(
color = 'rgb(49,130,189)',
opacity = 0.7
),
text = primary_sales,
textposition = 'auto',
)


# In[63]:


trace2 = go.Bar(
x = months,
y = secondary_sales,
name = 'Secondary Product',
marker = dict(
color = 'rgb(204,204,204)',
opacity = 0.5
),
text = secondary_sales,
textposition = 'auto',
)


# In[1]:


import plotly.graph_objs as go

langs = ['C', 'C++', 'Java', 'Python', 'PHP']
students = [23,17,35,29,12]

data = [go.Pie(
labels = langs,
values = students,
pull = [0.1,0,0,0,0]#나머지 파이보다 약간 떨어져있게 만듦
)]

fig = go.Figure(data=data)
fig.show()


# In[73]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots

countries = ["US", "China", "European Union", "Russian", "Federation", "Brazil", "India", "Rest of World"]
             
ghg = [16, 15, 12, 6, 5, 4, 42]
co2 = [27, 11, 25, 8, 1, 3, 25]


# In[71]:


fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

#fig.show()


# In[60]:


specs = [[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]]
fig = make_subplots(rows=2, cols=2, specs=specs)

fig.show()


# In[74]:


fig.add_trace(go.Pie(
labels=countries,
values=ghg,
name="GHG Emissions"),
row=1, col=1)

fig.add_trace(go.Pie(
labels=countries,
values=co2,
name="CO2 Emissions"),
row = 1, col = 2)


# In[80]:


fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(
title_text="Global Emissions 1990-2011",
annotations=[dict(text='GHG', x=0.18, y=0.5, font_size=20, showarrow=False),
dict(text='CO2', x=0.82, y=0.5, font_size=20, showarrow=False)])

fig.show()


# In[221]:


import plotly.graph_objs as go
import numpy as np

N = 100
x_vals = np.linspace(0, 1, N)

y1 = np.random.randn(N) + 5
y2 = np.random.randn(N)
y3 = np.random.randn(N) - 5


# In[222]:


trace0 = go.Scatter(
x = x_vals,
y = y1,
mode = 'markers',
name = 'markers'#,
#marker = dict(
#    color = 'rgb(49,130,189)',
#    opacity = 0.7,
#    size = 20)
)
trace1 = go.Scatter(
x = x_vals,
y = y2,
mode = 'lines+markers',
name = 'line+markers'
)
trace2 = go.Scatter(
x = x_vals,
y = y3,
mode = 'lines',
name = 'line'
)


# In[223]:


data = [trace0, trace1, trace2]
fig = go.Figure(data = data)
fig.show()


# In[88]:


import plotly.graph_objects as go

schools = ["Brown", "NYU", "Notre Dame", "Cornell", "Tufts", "Yale",
"Dartmouth", "Chicago", "Columbia", "Duke", "Georgetown",
"Princeton", "U.Penn", "Stanford", "MIT", "Harvard"]


# In[90]:


trace1 = go.Scatter(
x=[72, 67, 73, 80, 76, 79, 84, 78, 86, 93, 94, 90, 92, 96, 94, 112],
y=schools,
marker=dict(color="crimson", size=12),
mode="markers",
name="Women",
)

trace2 = go.Scatter(
x=[92, 94, 100, 107, 112, 114, 114, 118, 119, 124, 131, 137, 141, 151, 152, 165],
y=schools,
marker=dict(color="gold", size=12),
mode="markers",
name="Men",
)


# In[91]:


data = [trace1, trace2]

layout = go.Layout(title="Gender Earnings Disparity",
xaxis_title="Annual Salary (in thousands)",
yaxis_title="School")
fig = go.Figure(data=data, layout = layout)

fig.show()


# In[92]:


import plotly.graph_objects as go
import numpy as np

np.random.seed(1)

x = np.random.randn(500)

data = [go.Histogram(
x = x
)]

fig = go.Figure(data)
fig.show()


# In[95]:


import plotly.graph_objects as go

import numpy as np

x0 = np.random.randn(500)
x1 = np.random.randn(500) + 1

fig = go.Figure()
fig.add_trace(go.Histogram(x=x0))
fig.add_trace(go.Histogram(x=x1))

fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.show()


# In[96]:


import plotly.graph_objects as go
import pandas as pd

movies_df = pd.read_csv("C:/IMDB-Movie-Data.csv",index_col="Title")

x_values = movies_df['Rating']

mydata = go.Histogram(x=x_values)

mylayout = go.Layout(title='Rating frequencies',
xaxis_title = 'Rating',    
yaxis_title="Frequency")

fig = go.Figure(data = mydata, layout = mylayout)

fig.show()


# In[97]:


import plotly.graph_objects as go

import numpy as np
np.random.seed(1)

x = np.random.randn(500)
y = np.random.randn(500)+1

fig = go.Figure(go.Histogram2d(
x=x,
y=y
))
fig.show()


# In[99]:


import plotly.graph_objects as go

import numpy as np

movies_df = pd.read_csv("C:/IMDB-Movie-Data.csv",index_col="Title")

x = movies_df['Rating']
y = movies_df['Year']

fig = go.Figure(go.Histogram2d(
x=x,
y=y
))
fig.show()


# In[100]:


import plotly.graph_objects as go

yaxis = [1140,1460,489,594,502,508,370,200]

data = go.Box(y = yaxis)

fig = go.Figure(data)

fig.show()


# In[105]:


import plotly.graph_objects as go
import pandas as pd

movies_df = pd.read_csv("C:/IMDB-Movie-Data.csv",index_col="Title")

y_values = movies_df['Rating']

mydata = go.Box(y = y_values)

mylayout = go.Layout(title='Boxplot for Ratings', yaxis_title="Rating")

fig = go.Figure(mydata, mylayout)

fig.show()


# In[104]:


import plotly.graph_objects as go
import numpy as np

np.random.seed(10)
c1 = np.random.normal(100, 10, 200)
c2 = np.random.normal(80, 30, 200)

trace1 = go.Violin(y = c1, meanline_visible = True)
trace2 = go.Violin(y = c2, box_visible = True)

data = [trace1, trace2]
fig = go.Figure(data = data)
fig.show()


# In[5]:


import plotly.graph_objects as go
import pandas as pd

usedcars_df = pd.read_csv("C:/usedcars.csv")

x_values = usedcars_df['model']
y_values = x_values.value_counts()

print(y_values)

mydata = [go.Bar(
x = ['SE', 'SES', 'SEL'],
y = y_values
)]

fig = go.Figure(data = mydata)

fig.show()


# In[6]:


import plotly.graph_objects as go
import pandas as pd

usedcars_df = pd.read_csv("C:/usedcars.csv")

d = usedcars_df[['model', 'transmission']]
color = usedcars_df['transmission']

model = ['SE', 'SES', 'SEL']

#is_data1 = d['model']=='SEL'
#data1 = d[is_data1]


#print(data1)
#print(data1.value_counts())

#print(color.value_counts())

is_data1 = d['transmission']=='AUTO'
data1 = d[is_data1]

is_data2 = d['transmission']=='MANUAL'
data2 = d[is_data2]

auto = data1.value_counts()
manual = data2.value_counts()
print(auto)
print(manual)
#print()

trace1 = go.Bar(
x = model,
y = auto,
name = 'AUTO'
)

trace2 = go.Bar(
x = model,
y = manual,
name = 'MANUAL'
)

data = [trace1, trace2]
layout = go.Layout(barmode = 'group', title='model')
fig = go.Figure(data = data, layout = layout)
fig.show()


# In[9]:


import plotly.graph_objects as go
import pandas as pd

usedcars_df = pd.read_csv("C:/usedcars.csv")

d = usedcars_df[['model', 'transmission']]
color = usedcars_df['transmission']

model = ['SE', 'SES', 'SEL']

#is_data1 = d['model']=='SEL'
#data1 = d[is_data1]


#print(data1)
#print(data1.value_counts())

#print(color.value_counts())

is_data1 = d['transmission']=='AUTO'
data1 = d[is_data1]

is_data2 = d['transmission']=='MANUAL'
data2 = d[is_data2]

auto = data1.value_counts()
manual = data2.value_counts()
print(auto)
print(manual)
#print()

trace1 = go.Bar(
x = model,
y = auto,
name = 'AUTO',
    marker = dict(
    color = 'rgb(49,130,189)',
    opacity = 0.7)
)

trace2 = go.Bar(
x = model,
y = manual,
name = 'MANUAL',
    marker = dict(
    color = 'rgb(204,204,204)',
    opacity = 0.5)
)

data = [trace1, trace2]
layout = go.Layout(barmode = 'group', title='model')
fig = go.Figure(data = data, layout = layout)
fig.show()


# In[10]:


import plotly.graph_objects as go
import pandas as pd

usedcars_df = pd.read_csv("C:/usedcars.csv")

x_values = usedcars_df['model']
y_values = x_values.value_counts()

model = ['SE', 'SES', 'SEL']

print(y_values)

mydata = [go.Pie(
    labels = model,
    values = y_values,
    pull = [0,0,0.1]
)]

fig = go.Figure(data = mydata)

fig.show()


# In[12]:


import plotly.graph_objs as go
import pandas as pd

usedcars_df = pd.read_csv("C:/usedcars.csv")

year = usedcars_df['year'].values.tolist()
print(year)
price = usedcars_df['price'].values.tolist()
mileage = usedcars_df['mileage'].values.tolist()
print(price)
print(mileage)

trace1 = go.Scatter(
    x=year,
    y=price,
mode='lines')

trace2 = go.Scatter(
    x=year,
    y=mileage,
    mode='lines')

data = [trace1, trace2]
fig = go.Figure(data = data)
fig.show()


# In[13]:


import plotly.graph_objects as go
import pandas as pd

usedcars_df = pd.read_csv("C:/usedcars.csv",index_col="model")

x_values = usedcars_df['price']

mydata = go.Histogram(x=x_values)

mylayout = go.Layout(title='Price frequency',
xaxis_title = 'Price',    
yaxis_title="Frequency")

fig = go.Figure(data = mydata, layout = mylayout)

fig.show()


# In[14]:


import plotly.graph_objects as go

import numpy as np

usedcars_df = pd.read_csv("C:/usedcars.csv",index_col="model")

x = usedcars_df['price']
y = usedcars_df['year']

fig = go.Figure(go.Histogram2d(
x=x,
y=y
))
fig.show()


# In[15]:


import plotly.graph_objects as go
import pandas as pd

usedcars_df = pd.read_csv("C:/usedcars.csv",index_col="model")

x_values = usedcars_df['mileage']

mydata = go.Histogram(x=x_values)

mylayout = go.Layout(title='mileage frequency',
xaxis_title = 'mileage',    
yaxis_title="Frequency")

fig = go.Figure(data = mydata, layout = mylayout)

fig.show()


# In[16]:


import plotly.graph_objects as go

import numpy as np

usedcars_df = pd.read_csv("C:/usedcars.csv",index_col="model")

x = usedcars_df['mileage']
y = usedcars_df['year']

fig = go.Figure(go.Histogram2d(
x=x,
y=y
))
fig.show()


# In[17]:


import plotly.graph_objects as go
import pandas as pd

usedcars_df = pd.read_csv("C:/usedcars.csv",index_col="model")

y_values = usedcars_df['mileage']

mydata = go.Box(y = y_values)

mylayout = go.Layout(title='Boxplot for mileage', yaxis_title="mileage")

fig = go.Figure(mydata, mylayout)

fig.show()


# In[18]:


import plotly.graph_objects as go
import pandas as pd

usedcars_df = pd.read_csv("C:/usedcars.csv",index_col="model")
c1 = usedcars_df['mileage'].values.tolist()
c2 = usedcars_df['price'].values.tolist()
print(c1)
print(c2)

trace1 = go.Violin(y = c1, meanline_visible = True)
trace2 = go.Violin(y = c2, box_visible = True)

data = [trace1, trace2]
fig = go.Figure(data = data)
fig.show()


# In[19]:


import plotly.graph_objects as go
import numpy as np

np.random.seed(10)
c1 = np.random.normal(100, 10, 200)
c2 = np.random.normal(80, 30, 200)

print(c1)

trace1 = go.Violin(y = c1, meanline_visible = True)
trace2 = go.Violin(y = c2, box_visible = True)

data = [trace1, trace2]
fig = go.Figure(data = data)
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




