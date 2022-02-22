#!/usr/bin/env python
# coding: utf-8

# In[31]:


[10.5, 5.2, 3.25, 7.0]


# In[32]:


import numpy as np
video = np.array([10.5, 5.2, 3.25, 7.0])
video


# In[33]:


video.size


# In[34]:


video[2]  # 3rd element


# In[35]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[36]:


u = np.array([2, 5])
v = np.array([3, 1])


# In[37]:


x_coords, y_coords = zip(u, v)
#여러 개의 iterable 객체를 받은 후 자료형 들을 묶어서 튜플 형태로 출력해줌
plt.scatter(x_coords, y_coords, color=["r","b"])
plt.axis([0, 9, 0, 6])
plt.grid() #격자
plt.show()


# In[38]:


def plot_vector2d(vector2d, origin=[0, 0], **options):
    return plt.arrow(origin[0], origin[1], vector2d[0], vector2d[1],
              head_width=0.2, head_length=0.3, length_includes_head=True,
              **options)


# In[39]:


plot_vector2d(u, color="r")
plot_vector2d(v, color="b")
plt.axis([0, 9, 0, 6])
plt.grid()
plt.show()


# In[40]:


a = np.array([1, 2, 8])
b = np.array([5, 6, 3])


# In[41]:


from mpl_toolkits.mplot3d import Axes3D

subplot3d = plt.subplot(111, projection='3d') #?
x_coords, y_coords, z_coords = zip(a,b)
subplot3d.scatter(x_coords, y_coords, z_coords)
subplot3d.set_zlim3d([0, 9])
plt.show()


# In[42]:


def plot_vectors3d(ax, vectors3d, z0, **options):
    for v in vectors3d:
        x, y, z = v
        ax.plot([x,x], [y,y], [z0, z], color="gray", linestyle='dotted', marker=".")
    x_coords, y_coords, z_coords = zip(*vectors3d)
    ax.scatter(x_coords, y_coords, z_coords, **options)

subplot3d = plt.subplot(111, projection='3d')
subplot3d.set_zlim([0, 9])
plot_vectors3d(subplot3d, [a,b], 0, color=("r","b"))
plt.show()


# In[43]:


#Norm
#The norm of a vector  u , noted  ∥u∥ , is a measure of the length (a.k.a. the magnitude) of  u . There are multiple possible norms, but the most common one (and the only one we will discuss here) is the Euclidian norm, which is defined as:

#∥u∥=∑iui2−−−−−−√


# In[44]:


def vector_norm(vector):
    squares = [element**2 for element in vector]
    return sum(squares)**0.5

print("||", u, "|| =")
vector_norm(u)


# In[45]:


import numpy.linalg as LA
LA.norm(u)


# In[56]:


radius = LA.norm(u)
plt.gca().add_artist(plt.Circle((0,0), radius, color="#DDDDDD"))
plot_vector2d(u, color="red") # 왜안뜨지
plt.axis([0, 8.7, 0, 6])
plt.grid()
plt.show()


# In[57]:


print(" ", u)
print("+", v)
print("-"*10)
u + v


# In[58]:


plot_vector2d(u, color="r")
plot_vector2d(v, color="b")
plot_vector2d(v, origin=u, color="b", linestyle="dotted")
plot_vector2d(u, origin=v, color="r", linestyle="dotted")
plot_vector2d(u+v, color="g")
plt.axis([0, 9, 0, 7])
plt.text(0.7, 3, "u", color="r", fontsize=18)
plt.text(4, 3, "u", color="r", fontsize=18)
plt.text(1.8, 0.2, "v", color="b", fontsize=18)
plt.text(3.1, 5.6, "v", color="b", fontsize=18)
plt.text(2.4, 2.5, "u+v", color="g", fontsize=18)
plt.grid()
plt.show()


# In[65]:


t1 = np.array([2, 0.25])
t2 = np.array([2.5, 3.5])
t3 = np.array([1, 2])

x_coords, y_coords = zip(t1, t2, t3, t1)
plt.plot(x_coords, y_coords, "c--", x_coords, y_coords, "co") #점선, 점

plot_vector2d(v, t1, color="r", linestyle=":")
plot_vector2d(v, t2, color="r", linestyle=":")
plot_vector2d(v, t3, color="r", linestyle=":")

t1b = t1 + v
t2b = t2 + v
t3b = t3 + v

x_coords_b, y_coords_b = zip(t1b, t2b, t3b, t1b)
plt.plot(x_coords_b, y_coords_b, "b-", x_coords_b, y_coords_b, "bo") #점선, 점

plt.text(4, 4.2, "v", color="r", fontsize=18)
plt.text(3, 2.3, "v", color="r", fontsize=18)
plt.text(3.5, 0.4, "v", color="r", fontsize=18)

plt.axis([0, 6, 0, 5])
plt.grid()
plt.show()


# In[61]:


x_coords, y_coords


# In[ ]:





# In[ ]:




