#!/usr/bin/env python
# coding: utf-8

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path

import pandas as pd
import os
from sklearn import preprocessing
from sklearn.metrics import pairwise
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates


# In[14]:


df = pd.read_csv(r'C:\Users\dkris\Downloads\Country-data.csv')
df.set_index('country', inplace=True)
df


# In[17]:


df = df.apply(lambda x: x.astype('float64'))
df


# In[ ]:


t_df = df.drop(columns='country')


# In[18]:


kmeans = KMeans(n_clusters=7, random_state=0).fit(df)

# Cluster membership
memb = pd.Series(kmeans.labels_, index=df.index)
memb


# In[19]:


for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


# In[20]:


centroids = pd.DataFrame(kmeans.cluster_centers_, columns=df.columns)
pd.set_option('display.precision', 3)
print(centroids)
pd.set_option('display.precision', 6)

centroids['cluster'] = ['Cluster {}'.format(i) for i in centroids.index]
fig=plt.figure(figsize=(10,10),facecolor='white')
fig.subplots_adjust(right=3)
ax = parallel_coordinates(centroids, class_column='cluster', colormap='Dark2', linewidth=5)
plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.5))
plt.xlim(-0.5,7.5)


# In[24]:


# Normalized distance
df_norm = df.apply(preprocessing.scale, axis=0)

kmeans = KMeans(n_clusters=3, random_state=0).fit(df_norm)

# Cluster membership
memb = pd.Series(kmeans.labels_, index=df_norm.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


# In[25]:


centroids = pd.DataFrame(kmeans.cluster_centers_, columns=df.columns)
pd.set_option('display.precision', 3)
print(centroids)
pd.set_option('display.precision', 6)

centroids['cluster'] = ['Cluster {}'.format(i) for i in centroids.index]
fig=plt.figure(figsize=(10,10),facecolor='white')
fig.subplots_adjust(right=3)
ax = parallel_coordinates(centroids, class_column='cluster', colormap='Dark2', linewidth=5)
plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.5))
plt.xlim(-0.5,7.5)


# In[41]:


fig, ax = plt.subplots(facecolor='white')
inertia = []
for n_clusters in range(1, 10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df_norm)
    inertia.append(kmeans.inertia_ / n_clusters)
inertias = pd.DataFrame({'n_clusters': range(1, 10), 'inertia': inertia})
ax = inertias.plot(x='n_clusters', y='inertia', ax=ax)
plt.xlabel('Number of clusters(k)')
plt.ylabel('Average Within-Cluster Squared Distances')
plt.ylim((0, 1.1 * inertias.inertia.max()))
ax.legend().set_visible(False)
ax.patch.set_facecolor('white')
plt.show()


# In[36]:


centroids


# In[43]:


# Scatter plot for all clusters
plt.figure(figsize=(12, 8))

# Scatter plot for each cluster
for cluster_num in range(kmeans.n_clusters):
    cluster_points = df[memb == cluster_num]
    plt.scatter(cluster_points['Cluster'], cluster_points['Cluster'], label=f'Cluster {cluster_num}')

# Plotting centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')

plt.title('Scatter Plot for Clusters')
plt.xlabel('Feature_1')
plt.ylabel('Feature_2')
plt.legend()
plt.show()


# In[ ]:




