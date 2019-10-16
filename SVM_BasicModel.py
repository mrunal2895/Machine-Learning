#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Simple Support Vector Machine Model 


# In[48]:


#Dataset used is inbuilt wine dataset from sklearn


# In[49]:


from sklearn import datasets


# In[50]:


df=datasets.load_wine()


# In[51]:


df.feature_names


# In[52]:


df.target_names


# In[53]:


df.data


# In[54]:


df.data.shape


# In[55]:


df.target


# In[56]:


df.data[0:5]


# In[57]:


from sklearn.model_selection import train_test_split


# In[58]:


(X_train,X_test,y_train,y_test)=train_test_split(df.data,df.target,test_size=0.3,random_state=109)


# In[59]:


X_train.shape


# In[60]:


X_test.shape


# In[61]:


y_train.shape


# In[62]:


y_test.shape


# In[63]:


from sklearn import svm


# In[64]:


c=svm.SVC(kernel='linear')


# In[65]:


c.fit(X_train,y_train)


# In[66]:


z=c.predict(X_test)


# In[67]:


from sklearn import metrics
print("Accuracy of model is:",round(metrics.accuracy_score(y_test,z)*100),'%')


# In[ ]:





# In[ ]:




