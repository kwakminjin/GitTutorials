#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

dataset = [['Onion', 'Potato', 'Burger'],
['Potato', 'Burger', 'Milk'],
['Milk', 'Beer'],
['Potato', 'Milk'],
['Onion', 'Potato', 'Burger', 'Beer'],
['Onion', 'Potato', 'Burger', 'Milk']
]


# In[17]:


encode = TransactionEncoder()
encoded_array = encode.fit(dataset).transform(dataset)
encoded_array


# In[18]:


dataframe = pd.DataFrame(encoded_array, columns=encode.columns_)
dataframe


# In[28]:


frequent_itemsets = apriori(dataframe, min_support=0.4, use_colnames=True)
frequent_itemsets


# In[30]:


pattern_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
pattern_rules


# In[12]:


import pandas as pd
import seaborn as sns
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
basket = pd.read_csv("C:\Market_Basket_Optimisation.csv", header = None)
basket.head()


# In[ ]:


records = []
for i in range (0, 7501):
records.append([str(basket.values[i,j]) for j in range(0, 20)])


# In[ ]:


encode = TransactionEncoder()
encoded_array = encode.fit(records).transform(records)
data_frame = pd.DataFrame(encoded_array, columns = encode.columns_)
data_frame


# In[ ]:


basket_clean = data_frame.drop(['nan'], axis = 1)
basket_clean


# In[ ]:


frequent_itemsets = apriori(basket_clean, min_support = 0.04, use_colnames = True)
frequent_itemsets.head()


# In[ ]:


pattern_rules = association_rules(frequent_itemsets, metric = 'lift', min_threshold = 1)
pattern_rules


# In[ ]:





# In[ ]:





# In[ ]:




