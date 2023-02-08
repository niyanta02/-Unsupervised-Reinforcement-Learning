#!/usr/bin/env python
# coding: utf-8

# # Train a Gaussian mixture model on the Olivetti faces dataset.

# In[15]:


#importing libraries
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import IncrementalPCA
from sklearn.mixture import GaussianMixture


# In[16]:


from sklearn.datasets import fetch_olivetti_faces


# In[17]:


#dataset
face, target = fetch_olivetti_faces(data_home='./', return_X_y=True)
print(face[0, 0].dtype, target.dtype)
print(face.shape, target.shape)


# In[18]:


#feature reduction using PCA to reduce the dataset’s dimensionality
pca = IncrementalPCA(n_components=240)
face_reduced = pca.fit_transform(face)
print('Explained variance ratio in sum:')
print(sum(pca.explained_variance_ratio_))


# In[19]:


#decide the covariance type
gmm = GaussianMixture(n_components=2, covariance_type='spherical').fit(face_reduced)
labels = gmm.predict(face_reduced)
plt.scatter(face_reduced[:, 0], face_reduced[:, 100], c=labels)
plt.show()


# In[20]:


#search for the candidate k
#Determine the minimum number of clusters that best represent the dataset using either AIC or BIC.
bic_list = []
aic_list = []
search_space = range(1, 240, 10)
for cluster in search_space:
    gmm = GaussianMixture(n_components=cluster, covariance_type='spherical').fit(face_reduced)
    bic_list.append(abs(gmm.bic(face_reduced)))
    aic_list.append(abs(gmm.aic(face_reduced)))
#plotting
plt.plot(search_space, bic_list, label='BIC')
plt.plot(search_space, aic_list, label='AIC')
plt.xlabel('cluster')
plt.ylabel('information criterion')
plt.legend(loc='best')
plt.show()


# In[21]:


#Hard clustering for each instance
gmm = GaussianMixture(n_components=140, covariance_type='spherical').fit(face_reduced)
print('hard clustering')
print(gmm.predict(face_reduced))
#Soft clustering for each instance
print('soft clustering')
print(gmm.predict_proba(face_reduced))


# In[22]:


#Use the model to generate some new faces (using the sample() method), and visualize them (use the inverse_transform() method to transform the data back to its original space based on the PCA method used)
face_recovered = pca.inverse_transform(gmm.sample(2)[0])
origin = face_recovered.reshape(2, 64, 64)


# In[23]:


#images
plt.axis('off')
plt.imshow(origin[0])
plt.show()
plt.axis('off')
plt.imshow(origin[1])
plt.show()
rotate = np.array([origin[::-1][0].T, origin[::-1][1].T])
plt.axis('off')
plt.imshow(rotate[0])
plt.show()
plt.axis('off')
plt.imshow(rotate[1])
plt.show()


# In[24]:


origin = pca.transform(origin.reshape(2, 4096))
rotate = pca.transform(rotate.reshape(2, 4096))


# In[26]:


#Determine if the model can detect the anomalies produced in (8) by comparing the output of the score_samples() method for normal images and for anomalies)
print('score of origin')
print(gmm.score_samples(origin)) 
print('score of rotation')
print(gmm.score_samples(rotate))

