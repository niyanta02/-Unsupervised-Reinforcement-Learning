#!/usr/bin/env python
# coding: utf-8

# # Assignmnet 2 : KMeans Clustering

# In[55]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[56]:


#Retrieve and load the Olivetti faces dataset 

from sklearn.datasets import fetch_olivetti_faces
olivetti=fetch_olivetti_faces()


# In[57]:


print(olivetti.DESCR)


# In[58]:


#Split the training set, a validation set, and a test set using stratified sampling to ensure that there are the same 
#number of images per person in each set. Provide your rationale for the split ratio

from sklearn.model_selection import StratifiedShuffleSplit

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=27)
train_valid_idx, test_idx = next(strat_split.split(olivetti.data, olivetti.target))
X_train_valid = olivetti.data[train_valid_idx]
y_train_valid = olivetti.target[train_valid_idx]
X_test = olivetti.data[test_idx]
y_test = olivetti.target[test_idx]

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=43)
train_idx, valid_idx = next(strat_split.split(X_train_valid, y_train_valid))
X_train = X_train_valid[train_idx]
y_train = y_train_valid[train_idx]
X_valid = X_train_valid[valid_idx]
y_valid = y_train_valid[valid_idx]


# In[59]:


print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)


# In[60]:


target=olivetti.target
print(olivetti.target)


# In[61]:


#Using k-fold cross validation, train a classifier to predict which person is represented in each picture, and evaluate it 
#on the validation set

from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits = 4, shuffle = True, random_state =27)


# In[62]:


clf_lr=LogisticRegression(random_state=27,max_iter=10000)
clf=clf_lr.fit(X_train,y_train)


# In[64]:


score=np.mean(cross_val_score(clf,X_train, y_train,scoring="accuracy",cv=kfold))


# In[65]:


print(score)


# In[66]:


from sklearn.metrics import silhouette_score

silhouette_scores =silhouette_score(X_train, y_train)

print(silhouette_scores)


# In[67]:


classifier_RandFor= RandomForestClassifier()

pipeline = Pipeline([
    
    ('classifier_RaandFor', classifier_RandFor),
    
])

pipeline.fit(X_valid, y_valid)

crossVal_scores = cross_val_score(
    estimator=pipeline,
    X = X_valid,
    y = y_valid,
    scoring = 'accuracy',
    cv = kfold
)


print(crossVal_scores)


# In[68]:


#Use K-Means to reduce the dimensionality of the set. Provide your rationale for the similarity measure used to perform the \
#clustering. Use the silhouette score approach to choose the number of clusters.

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# In[69]:


k_range = range(5, 150, 5)
kmeans_per_k = []
for k in k_range:
    print("k={}".format(k))
    kmeans = KMeans(n_clusters=k, random_state=27).fit(X_train)
    kmeans_per_k.append(kmeans)


# In[70]:


pca = PCA(0.99)
X_train_pca = pca.fit_transform(X_train)
X_valid_pca = pca.transform(X_valid)
X_test_pca = pca.transform(X_test)


# In[71]:


pca.n_components_


# In[72]:


k_range = range(5, 150, 5)
kmeans_per_k = []
for k in k_range:
    print("k={}".format(k))
    kmeans = KMeans(n_clusters=k, random_state=59).fit(X_train_pca)
    kmeans_per_k.append(kmeans)


# In[73]:


from sklearn.metrics import silhouette_score
silhouette_scores = [silhouette_score(X_train, y_train)
                     for model in kmeans_per_k]
best_index = np.argmax(silhouette_scores)
best_k = k_range[best_index]
best_score = silhouette_scores[best_index]


# In[74]:


silhouette_scores = [silhouette_score(X_train_pca, model.labels_) 
                     for model in kmeans_per_k]
best_index = np.argmax(silhouette_scores)
best_k = k_range[best_index]
best_score = silhouette_scores[best_index]


# In[75]:


plt.figure(figsize=(8, 3))
plt.plot(k_range, silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.plot(best_k, best_score, "rs")
plt.show()


# In[76]:


print(best_k)


# In[77]:


inertias = [model.inertia_ for model in kmeans_per_k]
best_inertia = inertias[best_index]


# In[78]:


plt.figure(figsize=(8, 3.5))
plt.plot(k_range, inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.plot(best_k, best_inertia, "rs")
plt.show()


# In[79]:


best_model = kmeans_per_k[best_index]


# In[80]:


def plot_faces(faces, labels, n_cols=5):
    faces = faces.reshape(-1, 64, 64)
    n_rows = (len(faces) - 1) // n_cols + 1
    plt.figure(figsize=(n_cols, n_rows * 1.1))
    for index, (face, label) in enumerate(zip(faces, labels)):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(face, cmap="gray")
        plt.axis("off")
        plt.title(label)
    plt.show()


# In[81]:


for cluster_id in np.unique(best_model.labels_):
    print("Cluster", cluster_id)
    in_cluster = best_model.labels_==cluster_id
    faces = X_train[in_cluster]
    labels = y_train[in_cluster]
    plot_faces(faces, labels)


# In[82]:


#Continuing with the Olivetti faces dataset, train a classifier to predict which person is represented in each picture, and 
#evaluate it on the validation set.

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=150, random_state=27)
clf.fit(X_train_pca, y_train)
clf.score(X_valid_pca, y_valid)

X_train_reduced = best_model.transform(X_train_pca)
X_valid_reduced = best_model.transform(X_valid_pca)
X_test_reduced = best_model.transform(X_test_pca)

clf = RandomForestClassifier(n_estimators=150, random_state=27)
clf.fit(X_train_reduced, y_train)
    
clf.score(X_valid_reduced, y_valid)


# In[83]:


from sklearn.pipeline import Pipeline

for n_clusters in k_range:
    pipeline = Pipeline([
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=27)),
        ("forest_clf", RandomForestClassifier(n_estimators=150, random_state=27))
    ])
    pipeline.fit(X_train_pca, y_train)
    print(n_clusters, pipeline.score(X_valid_pca, y_valid))


# In[85]:


X_train_extended = np.c_[X_train_pca, X_train_reduced]
X_valid_extended = np.c_[X_valid_pca, X_valid_reduced]
X_test_extended = np.c_[X_test_pca, X_test_reduced]
clf = RandomForestClassifier(n_estimators=150, random_state=27)
clf.fit(X_train_extended, y_train)
clf.score(X_valid_extended, y_valid)

