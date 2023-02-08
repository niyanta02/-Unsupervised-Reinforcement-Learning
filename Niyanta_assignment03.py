#!/usr/bin/env python
# coding: utf-8

# # Assignment 3 Hierarchical Clustering
# 
# 

# In[7]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics


# In[2]:


#Retrieve and load the Olivetti faces dataset 

from sklearn.datasets import fetch_olivetti_faces
olivetti=fetch_olivetti_faces()


# In[3]:


X = olivetti.data
y = olivetti.target


# In[5]:


#Split the training set, a validation set, and a test set using stratified sampling to ensure that there are the same 
#number of images per person in each set. Provide your rationale for the split ratio

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=27)
train_valid_idx, test_valid_idx = next(strat_split.split(olivetti.data, olivetti.target))
X_train_valid = X[train_valid_idx]
y_train_valid = y[train_valid_idx]
X_test = X[test_valid_idx]
y_test = y[test_valid_idx]


strat_split = StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=27)
train_data_idx, valid_data_idx = next(strat_split.split(X_train_valid, y_train_valid))
X_train = X_train_valid[train_data_idx]
y_train = y_train_valid[train_data_idx]
X_valid = X_train_valid[valid_data_idx]
y_valid = y_train_valid[valid_data_idx]

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)


# In[8]:


#Classifier

clf = SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("accuracy score:{:.2f}%".format(clf.score(X_valid,y_valid)*100))

#plotting
import seaborn as sns
plt.figure(1, figsize=(16, 9))
sns.heatmap(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))


# In[9]:


#Using k-fold cross validation, train a classifier to predict which person is represented in each picture, 
#and evaluate it on the validation set. 

from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits=4, shuffle=True, random_state=27)
cv_scores=cross_val_score(clf, X, y, cv=kfold)
print("cross validations score for all 5 splits", cv_scores)
print("{} mean cross validations score:{:.2f}\n".format("", cv_scores.mean()))


# ## Using either Agglomerative Hierarchical Clustering (AHC) or Divisive Hierarchical Clustering (DHC) and using the complete linkage, reduce the dimensionality of the set by using the following similarity measures:
# - a) Euclidean Distance 
# - b) Minkowski Distance(p=1) or Manhattan Distance 
# - c) Cosine Similarity 

# In[12]:


from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_samples, silhouette_score


# In[13]:


agg = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
agg.fit(X_train)


# In[14]:


labels = agg.labels_


# In[15]:


plt.figure(figsize=(40,15))
dendro = sch.dendrogram(sch.linkage(X,"ward", metric="euclidean"))


# In[16]:


for k in range(30,150):
    hc_eu = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='average').fit(X)
    print(f"Silhouette Score for {k} is {silhouette_score(X, hc_eu.labels_, metric='euclidean')}")


# In[17]:


cluster_euc= AgglomerativeClustering(n_clusters=141, affinity='euclidean', linkage='average') 
cluster_euc_model = cluster_euc.fit(X)


# In[18]:


for k in range(30,150):
    hc_mink = AgglomerativeClustering(n_clusters=k, affinity='minkowski', linkage='average').fit(X)
    print(f"Silhouette Score for {k} is {silhouette_score(X, hc_mink.labels_, metric='minkowski')}")


# In[31]:


cluster_min= AgglomerativeClustering(n_clusters=141, affinity='minkowski', linkage='average') 
cluster_min_model = cluster_min.fit(X)


# In[32]:


for k in range(30,150):
    hc_cosine = AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='average').fit(X)
    print(f"Silhouette Score for {k} is {silhouette_score(X, hc_cosine.labels_, metric='cosine')}")


# In[33]:


cluster_cosine= AgglomerativeClustering(n_clusters=141, affinity='cosine', linkage='average') 
cluster_cosine_model = cluster_cosine.fit(X)


# In[46]:


#dimension reductionality


# In[38]:


from sklearn.cluster import FeatureAgglomeration

agg1 = FeatureAgglomeration(n_clusters=141, affinity='euclidean', linkage='average')
data_reduced_euc = agg1.fit_transform(X)

agg2 = FeatureAgglomeration(n_clusters=130, affinity='minkowski', linkage='average')
data_reduced_mink = agg2.fit_transform(X)

agg3 = FeatureAgglomeration(n_clusters=143, affinity='cosine', linkage='average')
data_reduced_cosine = agg3.fit_transform(X)


# In[47]:


#printing shapes 
print (euc.shape)
print (mink.shape)
print (cosine.shape)


# In[48]:


#train test and spit the reduced data
x_train_euc, x_test_euc, y_train_euc, y_test_euc = train_test_split(data_reduced_euc, y, stratify=y, test_size=.2)
x_train_mink, x_test_mink, y_train_mink, y_test_mink = train_test_split(data_reduced_mink, y, stratify=y, test_size=.2)
x_train_cos, x_test_cos, y_train_cos, y_test_cos = train_test_split(data_reduced_cosine, y, stratify=y, test_size=.2)


# In[42]:


cv = KFold(random_state=60, shuffle = True) 
model = SVC(kernel = "linear")
cv_ecu = cross_val_score(model, x_train_euc, y_train_euc , scoring='accuracy', cv=cv)
cv_mink = cross_val_score(model, x_train_mink, y_train_mink, scoring='accuracy', cv=cv)
cv_cosine = cross_val_score(model, x_train_cos, y_train_cos, scoring='accuracy', cv=cv)


# In[43]:


#accuracy score for euclidean

from sklearn.metrics import accuracy_score
model_ecu = model.fit(x_train_euc,y_train_euc)
y_predict_ecu = model_ecu.predict(x_test_euc)
print(accuracy_score(y_test_euc, y_predict_ecu))


# In[44]:


#accuracy score for minkowski

model_mink = model.fit(x_train_mink, y_train_mink)
y_predict_ecu = model_ecu.predict(x_test_mink)
print(accuracy_score(y_test_mink, y_predict_ecu))


# In[49]:


#accuracy score for cosine

model_cos = model.fit(x_train_cos, y_train_cos)
y_predict_ecu = model_ecu.predict(x_test_cos)
print(accuracy_score(y_test_cos, y_predict_ecu))

