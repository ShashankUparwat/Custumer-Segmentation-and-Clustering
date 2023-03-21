#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')


# In[12]:


df = pd.read_csv("/Users/shashankuparwat/Desktop/Customer Segmentation and Clustering/Customer.csv")


# In[13]:


df.head()


# df.info(

# In[14]:


df.info()


# # Univariate Analysis

# In[15]:


df.describe()


# In[16]:


sns.distplot(df['Annual Income (k$)'])


# In[17]:


df.columns


# In[19]:


cols = ['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']

for i in cols:
    plt.figure()
    sns.distplot(df[i])


# In[20]:


sns.kdeplot(df['Annual Income (k$)'], shade = True, hue = df['Gender'])


# In[21]:


cols = ['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']

for i in cols:
    plt.figure()
    sns.kdeplot(df[i], shade = True, hue = df['Gender'])


# In[22]:


cols = ['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']

for i in cols:
    plt.figure()
    sns.boxplot(data = df, x = 'Gender', y = df[i])


# In[26]:


df['Gender'].value_counts(normalize = True)


# # Bivariate Analysis

# In[28]:


sns.scatterplot(data = df, x = 'Annual Income (k$)', y = 'Spending Score (1-100)')


# In[31]:


#df = df.drop('CustomerID', axis = 1)
sns.pairplot(df, hue = 'Gender')


# In[32]:


df.groupby(['Gender'])['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# In[33]:


df.corr()


# In[34]:


sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')


# # Clustering - Univariate, Bivariate, Multivariate

# In[52]:


clustering1 = KMeans(n_clusters = 3)


# In[53]:


clustering1.fit(df[['Annual Income (k$)']])


# In[54]:


clustering1.labels_


# In[55]:


df['Income Cluster'] = clustering1.labels_
df.head()


# In[56]:


df['Income Cluster'].value_counts()


# In[57]:


clustering1.inertia_


# In[58]:


inertia_scores = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df[['Annual Income (k$)']])
    inertia_scores.append(kmeans.inertia_)


# In[59]:


inertia_scores


# In[60]:


plt.plot(range(1, 11),inertia_scores)


# In[61]:


df.columns


# In[63]:


df.groupby('Income Cluster')['Age', 'Annual Income (k$)', 'Spending Score (1-100)'].mean()


# In[64]:


#Bivariate Clustering


# In[70]:


clustering2 = KMeans(n_clusters = 5)
clustering2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])


# In[71]:


clustering2.labels_


# In[72]:


df['Spending and Income Cluster'] = clustering2.labels_
df.head()


# In[73]:


inertia_scores2 = []
for i in range(1, 11):
    kmeans2 = KMeans(n_clusters = i)
    kmeans2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    inertia_scores2.append(kmeans2.inertia_)


# In[74]:


plt.plot(range(1, 11),inertia_scores2)


# In[86]:


centers = pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x', 'y']
centers


# In[87]:


plt.figure(figsize = (10, 8))
plt.scatter(x = centers['x'], y = centers['y'], s = 100, c = 'black', marker = '*')
sns.scatterplot(data = df, x = 'Annual Income (k$)', y = 'Spending Score (1-100)', hue = 'Spending and Income Cluster', palette = 'tab10') 


# In[90]:


pd.crosstab(df['Spending and Income Cluster'], df['Gender'], normalize = 'index')


# In[91]:


df.groupby('Spending and Income Cluster')['Age', 'Annual Income (k$)', 'Spending Score (1-100)'].mean()


# In[92]:


#Multivariate Cluster


# In[93]:


from sklearn.preprocessing import StandardScaler


# In[94]:


scale = StandardScaler()


# In[95]:


df.head()


# In[98]:


dff = pd.get_dummies(df, drop_first = True)


# In[99]:


dff.head()


# In[100]:


dff.columns


# In[101]:


dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Male']]


# In[103]:


dff.head()


# In[108]:


dff = pd.DataFrame(scale.fit_transform(dff))


# In[109]:


dff


# In[117]:


inertia_scores3 = []
for i in range(1, 11):
    kmeans3 = KMeans(n_clusters = i)
    kmeans3.fit(dff)
    inertia_scores3.append(kmeans3.inertia_)


# In[118]:


plt.plot(range(1, 11), inertia_scores3)


# # Analysis

# ## Target Cluster
# 
# #### - Target group would be cluster 1 which has a high Spending Score and high income 
# 
# #### - 54 % of cluster 1 shoppers are women. We should look for ways to attract these customers using a marketing campaign targeting popular items in this cluster 
# 
# #### - Cluster 3 presents an interesting opportunity to market to the customers for sales event on popular items.

# In[ ]:




