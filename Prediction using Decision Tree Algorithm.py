#!/usr/bin/env python
# coding: utf-8

# ## AUTHOR- PAGARE MAITHILI
# ### DATA SCIENCE INTERN AT LET'S GROW MORE LGMVIP MAR23
# #### INTERMEDIATE LEVEL
# ### PREDICTION USING DECISION TREE ALGORITHM

# ## Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Reading The Data Set

# In[2]:


data=pd.read_csv('Iris.csv')
data.head()


# In[3]:


data.tail()


# ## Getting the Size of Data

# In[4]:


data.shape


# In[5]:


data.columns


# ## Checking for Null Values

# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# ## Getting some Statistical Inference from the Data

# In[8]:


data.describe(include='all')


# ## Data Visualization

# In[9]:


count = data['Species'].value_counts()
count.to_frame()


# In[10]:


label = count.index.tolist()
val = count.values.tolist()


# In[11]:


exp = (0.05,0.05,0.05)
fig,ax = plt.subplots()
ax.pie(val, explode=exp, labels=label, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Different Species of flower present in the Data")
ax.axis('equal')
plt.show()


# In[12]:


sns.pairplot(data=data, hue='Species')
plt.show()


# In[13]:


corr = data.corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm')


# ## Data Preparation

# In[14]:


data = data.drop('Id', axis=1)
data.head()


# In[15]:


x = data.iloc[:, 0:4]
x.head()


# In[16]:


y = (data.iloc[:, 4])
y.head().to_frame()


# In[17]:


from sklearn.preprocessing import StandardScaler


# In[18]:


std = StandardScaler()
x = std.fit_transform(x)


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 1)


# ## Model Creation

# In[21]:


from sklearn.tree import DecisionTreeClassifier


# In[22]:


model = DecisionTreeClassifier()
model.fit(x_train, y_train)


# ## Prediction using the Created Model

# In[23]:


y_pred = model.predict(x_test)
y_pred


# ## Model Evaluation

# In[24]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[25]:


acc = accuracy_score(y_test, y_pred)
print("The Accuracy of the Decision Tree Algorithms is : ", str(acc*100) + "%")


# In[26]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[27]:


lst = data['Species'].unique().tolist()
df_cm = pd.DataFrame(data = cm, index = lst, columns = lst)
df_cm


# ## Data Visualization for the Model

# In[28]:


data.columns


# In[29]:


col = data.columns.tolist()
print(col)


# In[30]:


from sklearn.tree import plot_tree


# In[31]:


fig = plt.figure(figsize=(25, 20))
tree_img = plot_tree(model, feature_names = col, class_names = lst, filled = True)

