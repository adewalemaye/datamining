
# %% [markdown]
# # HW Numpy 
# ## By: Adewale Maye
# ### Date: 02/14/2023

# %%
#%%
# NumPy

import numpy as np

# %% [markdown]
# ## Question 1

# %%
list2 = [ [11,12,13], [21,22,23], [31,32,33], [41,42,43] ] # two dimensional list (2-D array)  # (4,3)
nparray2 = np.array(list2)
print("nparray2:", nparray2)
# For details on indices function, see class notes or Class05_02_NumpyCont.py
idlabels = np.indices( (4,3) ) 
print("idlabels:", idlabels)

i,j = idlabels  # idlabels is a tuple of length 2. We'll call those i and j
nparray2b = 10*i+j+11
print("nparray2b:",nparray2b)

# %% [markdown]
# ### Question 1a

# %%
print(nparray2 is nparray2b)

# %%
print(nparray2 == nparray2b)

# %% [markdown]
# #### The "is" function shows that both arrays are referring to different objects in this case. With the "==" operator, we see that the values of the array are the same across the board.

# %% [markdown]
# ### Question 1b

# %%
# Question 1b - Object Type
print(type(idlabels))
print(type(i))
print(type(j))
# Shape
print(idlabels.shape)
print(i.shape)
print(j.shape)
# data types
print(idlabels.dtype)
print(i.dtype)
print(j.dtype)
# strides
print(idlabels.strides)
print(i.strides)
print(j.strides)

# %% [markdown]
# ### Question 1c

# %%
print(i)
print(idlabels)

# %%
i[0,0] = 8
print(i)
print(idlabels)

# %% [markdown]
# #### The first object of the array is changed to 8 in "i", and it is also reflected in the idlabels variable. Yes, this is expected.

# %%
i[0] = 8
print(i)
print(idlabels)

# %% [markdown]
# ### Question 1d

# %%
nparray2

nparray2c = nparray2.view()
nparray2c[1,1] = 0

print(nparray2)
print(nparray2c)

# Since this is a shallow copy (using view), this is completely expected. The change made in the view is reflected in the original .
print(nparray2 is nparray2c)

print(nparray2 == nparray2c)

# %% [markdown]
# ### Question 1e

# %%
nparray2 = np.array(list2) 
nparray2c = nparray2.copy()

nparray2c[0,2] = -1

print(nparray2)
print(nparray2c) 

# at this point, they are not true copies. The change in the copy is not reflected in the original.

print(nparray2 is nparray2c)

print(nparray2 == nparray2c)

# %% [markdown]
# ## Question 2

# %%
print(np.allclose([1e10,1e-7], [1.00001e10,1e-8]))
print(np.allclose([1e10,1e-8], [1.00001e10,1e-9]))
print(np.allclose([1e10,1e-8], [1.0001e10,1e-9]))

# %% [markdown]
# ## Question 3

# %%
x = np.arange(12, 38)
x
new = np.flip(x)
print(new)

# %% [markdown]
# ## Question 4

# %%
a = np.ones((7,7), dtype='int32')

b = np.zeros((5,5), dtype = 'int32')

a[1:-1,1:-1] = b
print(a)

# %% [markdown]
# ## Question 5

# %%
i=3642
myarray = np.arange(i,i+6*11).reshape(6,11)
#print(myarray)


#Boolean Matrix
print(myarray % 7 == 0)

# multiples of 7 within array
multiples_7 = myarray[myarray%7 == 0]
print(multiples_7)

# %% [markdown]
# ## Question 6

# %%
flatlist = list(range(1,25))
print(flatlist)

# %% [markdown]
# ### Question 6.1

# %%
nparray1 = np.array(flatlist)
print(nparray1.shape)

# %% [markdown]
# ### Question 6.2

# %%
nparray2 = nparray1.reshape(3,8)
print(nparray2)

# %% [markdown]
# ### Question 6.3

# %%
nparray3 = nparray2
nparray3[:,[0,2]] = nparray2[:,[2,0]]
print(nparray3)

# %% [markdown]
# ### Question 6.4

# %%
print(nparray3)
nparray4 = nparray3
nparray4[[0,1],:] = nparray3[[1,0],:] 
print(nparray4)

# %% [markdown]
# ### Question 6.5

# %%
nparray3D = nparray4.reshape(2,3,4)
print(nparray3D)

# %% [markdown]
# ### Question 6.6

# %%
nparray5 = (nparray3D % 3 == 0)
print(nparray5)

# %% [markdown]
# ### Question 6.7

# %%
# multiples of 3 within array
multiples_3 = nparray3D[nparray3D % 3 == 0]
print(multiples_3)

# shape of new array
print(multiples_3.shape)

# %% [markdown]
# ### Question 6.8

# %%
print(nparray3D)

div_3 = nparray3D % 3 == 0

nparray6b = np.where(div_3,nparray3D, 0)
print(nparray6b)


