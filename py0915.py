#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
print(np.__version__)


# In[3]:


a = np.array([1, 2, 3])
print(a)


# In[4]:


arr = np.array(42)

print(arr)


# In[5]:


arr = np.array([1, 2, 3, 4, 5])

print(arr)


# In[6]:


arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr)


# In[9]:


arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(arr)


# In[11]:


np.zeros(2)


# In[12]:


np.ones(2)


# In[13]:


np.empty(3)


# In[14]:


np.full(3, 7)


# In[15]:


np.full(shape = 3, fill_value = 7)


# In[16]:


np.eye(3)


# In[17]:


np.random.random((2,2))


# In[18]:


np.arange(4)


# In[19]:


np.arange(2, 9, 2)


# In[20]:


np.linspace(0, 10, num=5)


# In[21]:


srr = np.array([0, 1, 2, 3])
print(arr.dtype)


# In[22]:


arr = np.array([1, 2, 3, 4], dtype='f')
print(arr)
print(arr.dtype)


# In[33]:


arr = np.array(['a', 2, 3], dtype='i')
print(arr)

##print(arr.dtype)


# In[34]:


arr = np.ones(5, dtype='i')
print(arr)


# In[36]:


arr = np.zeros(5, dtype='bool')
print(arr)


# In[118]:


arr = np.array([1.1, 2.1, 3.1])

newarr = arr.astype('i')

print(newarr)
print(newarr.dtype)


# In[120]:


arr = np.array([1, 0 , 3])

newarr = arr.astype('bool')

print(newarr)
print(newarr.dtype)


# In[40]:


array_example = np.array([[[0, 1, 2, 3],
                          [4, 5, 6, 7]],
                          [[0, 1, 2, 3],
                          [4, 5, 6, 7]],
                          [[0, 1, 2, 3],
                          [4, 5, 6, 7]]])

print(array_example.ndim)


# In[41]:


array_example = np.array([[[0, 1, 2, 3],
                          [4, 5, 6, 7]],
                          [[0, 1, 2, 3],
                          [4, 5, 6, 7]],
                          [[0, 1, 2, 3],
                          [4, 5, 6, 7]]])

print(array_example.size)


# In[45]:


array_example = np.array([[[0, 1, 2, 3],
                          [4, 5, 6, 7]],
                          [[0, 1, 2, 3],
                          [4, 5, 6, 7]],
                          [[0, 1, 2, 3],
                          [4, 5, 6, 7]]])

print(len(array_example[0, 0]))


# In[49]:


array_example = np.array([[[0, 1, 2, 3],
                          [4, 5, 6, 7]],
                          [[0, 1, 2, 3],
                          [4, 5, 6, 7]],
                          [[0, 1, 2, 3],
                          [4, 5, 6, 7]]])

print(array_example.shape)


# In[51]:


arr = np.array([[[1, 2, 3],
                 [4, 5, 6]],
                [[7, 8, 9],
                [10, 11, 12]]])

print(arr[0, 1, 2])


# In[52]:


a = np.arange(6)

b = a.reshape(3, 2)

print(b)


# In[69]:


arr = np.array([[1, 2, 3], [4, 5, 6]])

newarr = arr.reshape(-1)

print(newarr)


# In[123]:


arr = np.arange(1, 13, 1)

print(arr)

newarr = arr.reshape(2, 2, 3)

print(newarr)


# In[83]:


data = np.array([1, 2, 3])

print(data[1])
print(data[0:2])
print(data[1:])
print(data[-2:])


# In[84]:


arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[-3:-1])


# In[85]:


arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[0:2, 2])


# In[87]:


arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.concatenate((arr1, arr2))

print(arr)


# In[98]:


arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.stack((arr1, arr2), axis=1)

print(arr)


# In[99]:


arr = np.array([1, 2, 3, 4, 5, 6])

newarr = np.array_split(arr, 3)

print(newarr)


# In[100]:


arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42

print(arr)
print(x)


# In[124]:


arr = np.array([1, 2, 3, 4, 5, 4, 4])

x = np.where(arr == 4)

print(x)


# In[116]:


arr1 = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])

arr2 = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])

x = np.where(arr1 == arr2)

print(x)


# In[103]:


a = np.arange(15)

index = np.where((a >= 5) & (a <= 10))

print(a[index])


# In[104]:


a = np.array([1, 2, 3, 2, 3, 4, 3, 5, 6])
b = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])

np.intersect1d(a, b)


# In[105]:


a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 6, 7, 8, 9])

np.setdiff1d(a, b)


# In[106]:


arr = np.array([3, 2, 0, 1])

print(np.sort(arr))


# In[107]:


a = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])

unique_values = np.unique(a)

print(unique_values)


# In[110]:


a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [10, 11, 12], [7, 8, 9]])

unique_values = np.unique(a)

print(unique_values)


# In[111]:


data = np.array([1, 2])
ones = np.ones(2, dtype=int)

print(data + ones)


# In[113]:


print(data - ones)
print(data * data)
print(data / data)


# In[114]:


data = np.array([1.0, 2.0])

print(data * 1.6)


# In[115]:


print(data.max())
print(data.min())
print(data.sum())

