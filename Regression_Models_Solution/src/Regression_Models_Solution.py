#!/usr/bin/env python
# coding: utf-8

# # Model Training

# ### scikit - learn
# 
# https://scikit-learn.org/stable/
# 
# scikitlearn (sklearn) provides simple and efficient tools for predictive data analysis. It is built on NumPy, SciPy, and matplotlib. 

# First thing, Import all the libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', 50)


# In[2]:


# next load the data
df = pd.read_csv('final.csv')
df.head()


# In[5]:


df.shape


# ## Linear Regression Model

# In[3]:


# import linear regression model
from sklearn.linear_model import LinearRegression


# In[4]:


# seperate input features in x
x = df.drop('price', axis=1)

# store the target variable in y
y = df['price']


# **Train Test Split**
# * Training sets are used to fit and tune your models.
# * Test sets are put aside as "unseen" data to evaluate your models.
# * The `train_test_split()` function splits data into randomized subsets.

# In[5]:


# import module
from sklearn.model_selection import train_test_split

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1234)


# In[6]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[7]:


# train your model
lrmodel = LinearRegression().fit(x_train,y_train)

# make preditions on train set
train_pred = lrmodel.predict(x_train)


# In[9]:


# evaluate your model
# we need mean absolute error
from sklearn.metrics import mean_absolute_error

train_mae = mean_absolute_error(train_pred, y_train)
print('Train error is', train_mae)


# In[23]:


lrmodel.coef_


# In[24]:


lrmodel.intercept_


# In[11]:


# make predictions om test set
ypred = lrmodel.predict(x_test)

#evaluate the model
test_mae = mean_absolute_error(ypred, y_test)
print('Test error is', test_mae)


# ### Our model is still not good beacuse we need a model with Mean Absolute Error < $70,000
# 
# Note - We have not scaled the features and not tuned the model.

#     

# ## Decision Tree Model

# In[42]:


# import decision tree model
from sklearn.tree import DecisionTreeRegressor


# In[49]:


# create an instance of the class
dt = DecisionTreeRegressor(max_depth=3, max_features=10, random_state=567)


# In[50]:


# train the model
dtmodel = dt.fit(x_train,y_train)


# In[51]:


# make predictions using the test set
ytest_pred = dtmodel.predict(x_test)


# In[52]:


# evaluate the model
test_mae = mean_absolute_error(ytest_pred, y_test)
test_mae


# ## How do I know if my model is Overfitting or Generalised?

# In[53]:


# make predictions on train set
ytrain_pred = dtmodel.predict(x_train)


# In[54]:


# import mean absolute error metric
from sklearn.metrics import mean_absolute_error

# evaluate the model
train_mae = mean_absolute_error(ytrain_pred, y_train)
train_mae


#     

# ## Plot the tree

# In[80]:


# get the features
dtmodel.feature_names_in_


# In[83]:


# plot the tree
from sklearn import tree

# Plot the tree with feature names
tree.plot_tree(dtmodel, feature_names=dtmodel.feature_names_in_)

#tree.plot_tree(dtmodel)
#plt.show(dpi=300)

# Save the plot to a file
plt.savefig('tree.png', dpi=300)


#     

#     

#     

# ## Random Forest Model

# In[85]:


# import decision tree model
from sklearn.ensemble import RandomForestRegressor


# In[86]:


# create an instance of the model
rf = RandomForestRegressor(n_estimators=200, criterion='absolute_error')


# In[87]:


# train the model
rfmodel = rf.fit(x_train,y_train)


# In[88]:


# make prediction on train set
ytrain_pred = rfmodel.predict(x_train)


# In[89]:


# make predictions on the x_test values
ytest_pred = rfmodel.predict(x_test)


# In[90]:


# evaluate the model
test_mae = mean_absolute_error(ytest_pred, y_test)
test_mae


# In[94]:


# Individual Decision Trees
# tree.plot_tree(rfmodel.estimators_[2], feature_names=dtmodel.feature_names_in_)


#     

# ## Pickle: 
# 
# * The pickle module implements a powerful algorithm for serializing and de-serializing a Python object structure. 
# 
# * The saving of data is called Serialization, and loading the data is called De-serialization.
# 
# **Pickle** model provides the following functions:
# * **`pickle.dump`** to serialize an object hierarchy, you simply use `dump()`. 
# * **`pickle.load`** to deserialize a data stream, you call the `loads()` function.

# In[95]:


# import pickle to save model
import pickle
 
# Save the trained model on the drive 
pickle.dump(rfmodel, open('RE_Model','wb'))


# In[96]:


# Load the pickled model
RE_Model = pickle.load(open('RE_Model','rb'))

np.array(xtrain.loc[22])
ytrain[22]
# In[99]:


# Use the loaded pickled model to make predictions
RE_Model.predict([[2012, 216, 74, 1 , 1, 618, 2000, 600, 1, 0, 0, 6, 0]])


# In[ ]:




