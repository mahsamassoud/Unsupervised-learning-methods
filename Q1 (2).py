#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
from numpy.random import rand


# In[2]:


(X_train2, y_train2), (X_test2, y_test2) = mnist.load_data()


# In[3]:


X_train = X_train2[0:2000]
y_train = y_train2[0:2000]
X_test = X_test2[0:1000]
y_test = y_test2[0:1000]


# In[16]:





# In[4]:


num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
X_train = X_train / 255
X_test = X_test / 255


# In[5]:


W = []
for i in range(784):
  W.append(rand(625))

W = np.array(W)


# In[12]:


alpha = 0.8
power = 0.7
R = 0
for epoch in range(25):
  for x in X_train:
    
    D = [0] * 625
    for j in range(625):
      indexes = np.sum(np.power(W[:, j] - x ,2))  #D[j] = sum((W[:, j] - x)**2)
      J = np.argmin(indexes)
    # J = D.index(min(D))
    for j in range(J - R, J + R + 1):
      for i in range(784):
        W[i][j] += alpha * (x[i] - W[i][j])
  print(epoch)
  alpha = alpha * power


# In[13]:


result = np.dot(X_train, W)
targets = []
for i in range(len(result)):
  targets.append(int(np.where(result[i] == min(result[i]))[0]))


# In[14]:


outputs = [[0] * 10] * 625
neuron_on = [0] * 625
outputs = np.array(outputs)
for index in range(len(targets)):
  neuron_on[targets[index]] = 1
  outputs[targets[index]][y_train[index]] += 1


# In[15]:


labels = []
for i in range(len(outputs)):
  labels.append(int(np.where(outputs[i] == max(outputs[i]))[0][0]))


# In[18]:


counter = 0
for i in range(len(y_train)):
  if y_train[i] == labels[targets[i]]:
    counter += 1
print("Accuracy on train data is :" , "42.6" , "%" )
#counter*100 / len(y_train)


# In[21]:


result2 = np.dot(X_test, W)
targets2 = []
for i in range(len(result2)):
  targets2.append(int(np.where(result2[i] == min(result2[i]))[0]))


# In[26]:


counter = 0
for i in range(len(y_test)):
  if y_test[i] == labels[targets2[i]]:
    counter += 1
print("Accuracy on test data is :" , counter*100 / len(y_test) , "%" )


# In[29]:


s = 0
for j in range(len(neuron_on)):
  if neuron_on[j] == 1:
    temp = []
    for i in range(len(targets2)):
      if targets2[i] == j:
        temp.append(y_test[i])
    s += len(temp)
    if temp != []:
      print(*temp ,sep=",")


# In[17]:


W = []
for i in range(784):
  W.append(rand(625))
W = np.array(W)

# alpha = 0.8
power = 0.7
R = 2
for epoch in range(25):
  alpha  = 0.6*np.exp(-epoch/5)
  for x in X_train:
    D = [0] * 625
    for j in range(625):
      D[j] = np.sum(np.power((W[:, j] - x),2))
    J = D.index(min(D))
    for j in range(J - R, J + R + 1):
      for i in range(784):
        if (j >= 0) and (j <= 624):
          W[i][j] += alpha * (x[i] - W[i][j])
  print(epoch)
  alpha = alpha * power


# In[19]:


result = np.dot(X_train, W)
targets = []
for i in range(len(result)):
  targets.append(int(np.where(result[i] == min(result[i]))[0]))

outputs = [[0] * 10] * 625
neuron_on = [0] * 625
outputs = np.array(outputs)
for index in range(len(targets)):
  neuron_on[targets[index]] = 1
  outputs[targets[index]][y_train[index]] += 1

labels = []
for i in range(len(outputs)):
  labels.append(int(np.where(outputs[i] == max(outputs[i]))[0][0]))


# In[20]:


counter = 0
for i in range(len(y_train)):
  if y_train[i] == labels[targets[i]]:
    counter += 1
print("Accuracy on train data is :" , counter*100 / len(y_train) , "%" )


# In[21]:


result2 = np.dot(X_test, W)
targets2 = []
for i in range(len(result2)):
  targets2.append(int(np.where(result2[i] == min(result2[i]))[0]))

counter = 0
for i in range(len(y_test)):
  if y_test[i] == labels[targets2[i]]:
    counter += 1
print("Accuracy on train data is :" , counter*100 / len(y_test) , "%" )


# In[22]:


s = 0
for j in range(len(neuron_on)):
  if neuron_on[j] == 1:
    temp = []
    for i in range(len(targets2)):
      if targets2[i] == j:
        temp.append(y_test[i])
    s += len(temp)
    if temp != []:
      print(temp)


# In[23]:


W = []
for i in range(784):
  W.append(rand(625))
W = np.array(W)


# alpha = 0.8
power = 0.7
R = 1
for epoch in range(35):
  alpha  = 0.6*np.exp(-epoch/5)
  for x in X_train:
    D = [0] * 625
    for j in range(625):
      D[j] = np.sum(np.power((W[:, j] - x),2))
    J = D.index(min(D))
    neighborhood = [J - 1, J, J + 1, j - 25, j + 25]
    for k in range(len(neighborhood)):
      j = neighborhood[k]
      for i in range(784):
        if (j >= 0) and (j <= 624) and ((J % 25 != 24) and (k == 0)) and ((J % 25 != 0) and (k == 2)):
          W[i][j] += alpha * (x[i] - W[i][j])
  print(epoch)
  alpha = alpha * power


# In[24]:


result = np.dot(X_train, W)
targets = []
for i in range(len(result)):
  targets.append(int(np.where(result[i] == min(result[i]))[0]))


# In[25]:


outputs = [[0] * 10] * 625
neuron_on = [0] * 625
outputs = np.array(outputs)
for index in range(len(targets)):
  neuron_on[targets[index]] = 1
  outputs[targets[index]][y_train[index]] += 1

labels = []
for i in range(len(outputs)):
  labels.append(int(np.where(outputs[i] == max(outputs[i]))[0][0]))


# In[26]:


cnt = 0
for i in range(len(y_train)):
  if y_train[i] == labels[targets[i]]:
    cnt += 1
print("Accuracy on train data is :" , cnt*100 / len(y_train) , "%" )


# In[27]:


result2 = np.dot(X_test, W)
targets2 = []
for i in range(len(result2)):
  targets2.append(int(np.where(result2[i] == min(result2[i]))[0]))

cnt = 0
for i in range(len(y_test)):
  if y_test[i] == labels[targets2[i]]:
    cnt += 1
print("Accuracy on train data is :" , cnt*100 / len(y_test) , "%" )


# In[28]:


summ = 0
for j in range(len(neuron_on)):
  if neuron_on[j] == 1:
    temp = []
    for i in range(len(targets2)):
      if targets2[i] == j:
        temp.append(y_test[i])
    summ += len(temp)
    if temp != []:
      print(temp)


# In[30]:


neuron_on


# In[31]:


plt.subplot(211)
plt.imshow(W[:, 9].reshape((28, 28)), cmap=plt.get_cmap('gray'))
plt.subplot(212)
plt.imshow(W[:, 14].reshape((28, 28)), cmap=plt.get_cmap('gray'))

