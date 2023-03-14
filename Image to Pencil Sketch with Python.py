#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


image = cv2.imread('./dog.webp')


# In[3]:


cv2.imshow("Original Image", image)
cv2.waitKey(0)


# In[4]:


b_and_w_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.waitKey(0)


# In[5]:


cv2.imshow("Black and White Image", b_and_w_image)
cv2.waitKey(0)


# In[6]:


inverted_image = 255 - b_and_w_image


# In[7]:


cv2.imshow("Inverted Image", inverted_image)
cv2.waitKey()


# In[8]:


blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)


# In[9]:


inverted_blurred = 255 - blurred


# In[10]:


pencil_sketch = cv2.divide(b_and_w_image, inverted_blurred, scale=256.0)


# In[11]:


cv2.imshow("Sketch Image", pencil_sketch)
cv2.waitKey(0)


# In[12]:


cv2.imshow("Original Image", image)


# In[13]:


cv2.imshow("Pencil Sketch Image", pencil_sketch)
cv2.waitKey(0)

