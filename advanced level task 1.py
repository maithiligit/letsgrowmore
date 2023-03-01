#!/usr/bin/env python
# coding: utf-8

# ## AUTHOR - MAITHILI PAGARE
# ### DATA SCIENCE INTERN AT LET'S GROW MORE LGMVIP MARCH 23
# ### Advanced Level Task
# ### Develop A Neural Network That Can Read Handwriting
# ### Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras


# ### Loading the dataset

# In[2]:


mnist= tf.keras.datasets.mnist
     


# ### Dividing the images and their outputs into x_train and x_test sets 

# In[3]:


(x_train,y_train),(x_test,y_test)=mnist.load_data()


# ### Cheking the data set images, which needed to be identified by the neural network

# In[4]:


fig=plt.figure(figsize=(15,3))
for i in range(20):
  ax=fig.add_subplot(2,10,i+1)
  ax.imshow(np.squeeze(x_train[i]),cmap='Reds')
  ax.set_title(y_train[i])


# ### Printing the data images into metrics form

# In[5]:


print(x_train.shape)
print(x_train[0])


# ### Normalizing the matrix array of number images
# 

# In[6]:


xtrain = x_train/255.0
xtest = x_test/255.0


# ### Flattening the 2-dimensional array into one dimensional array or a single column; which will behave as 1st inpur layer for neural network

# In[7]:


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                    tf.keras.layers.Dense(128, activation ='relu'),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
     


# In[8]:


model.summary()


# ### Using of AdamOptimizer

# In[9]:


model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss ='sparse_categorical_crossentropy',
            metrics=['accuracy'])


# ### Fitting of the training data into the model for 5 iterations

# In[10]:



model.fit(xtrain,y_train, epochs=5)


# ### 98% Accuracy achieved with 5 iterations
# ### Fitting of the training data into the model for 9 iterations

# In[11]:


model.fit(xtrain,y_train, epochs=9)


# ### 99% Accuracy achieved with 9 iterations

# In[12]:


print(model.evaluate(x_test,y_test))


# ### 97 % of total accuracy of our neural network has been achieved

# In[13]:


history=model.fit(x_train,y_train,epochs=10,batch_size=32,validation_split=0.1)


# ### Plotting Accuracy of Model

# In[14]:


plt.title("Accuracy and Loss")
plt.xlabel("Epoch")
plt.ylabel("acc/Loss")
plt.plot(history.history['accuracy'],label='acc')
plt.plot(history.history["val_accuracy"],label='val')
plt.show()    


# In[15]:


plt.title("Accuracy and Loss")
plt.xlabel("Epoch")
plt.ylabel("acc/Loss")
plt.plot(history.history['loss'],label='acc')
plt.plot(history.history["val_loss"],label='val')
plt.show()
     


# ### Testing Our Model

# In[16]:


plt.imshow(np.squeeze(x_test[0]),cmap="Reds")


# In[17]:


prediction=model.predict(x_test)
print(np.argmax(prediction[0]))


# In[18]:


plt.imshow(np.squeeze(x_test[1]),cmap="Reds")


# In[19]:


prediction=model.predict(x_test)
print(np.argmax(prediction[1]))


# ## Thank You
