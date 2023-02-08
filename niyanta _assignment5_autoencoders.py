#!/usr/bin/env python
# coding: utf-8

# # Niyanta Assignment-05 Autoencoders

# In[4]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from keras.layers import Dense, Input
from keras.models import Model


# In[6]:


#fetching dataset
X_data, y_target = fetch_olivetti_faces(return_X_y=True, random_state=27)


# In[7]:


# applying stratified sampling to split the data into test, train and validation

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=27)
train_valid_idx, test_idx = next(strat_split.split(X_data, y_target))
X_train_valid = X_data[train_valid_idx]
y_train_valid = y_target[train_valid_idx]
X_test = X_data[test_idx]
y_test = y_target[test_idx]

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=27)
train_idx, valid_idx = next(strat_split.split(X_train_valid, y_train_valid))
X_train = X_train_valid[train_idx]
y_train = y_train_valid[train_idx]
X_valid = X_train_valid[valid_idx]
y_valid = y_train_valid[valid_idx]


# In[9]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_valid.shape, y_valid.shape)


# In[10]:


#dimensionality reduction using PCA

pca = PCA(0.99)
X_train_reduced = pca.fit_transform(X_train)
X_valid_reduced = pca.transform(X_valid)
X_test_reduced = pca.transform(X_test)
pca.n_components_


# In[11]:


#applying inverse transformation
X_train_reconstructed = pca.inverse_transform(X_train_reduced)
X_valid_reconstructed = pca.inverse_transform(X_valid_reduced)
X_test_reconstructed = pca.inverse_transform(X_test_reduced)


# In[12]:


#printing final shape
X_train_reconstructed.shape


# In[13]:


# building model architecture function, will return encoder model object

def build_model(input_size, hidden_size, code_size):
    input_img = Input(shape=(input_size,))

    hidden1= Dense(hidden_size,activation='relu')(input_img)
    code= Dense(code_size,activation='relu')(hidden1)
    hidden2 = Dense(hidden_size,activation='relu')(code)

    output_img = Dense(input_size,activation='sigmoid')(hidden2)
    return Model(input_img,output_img)


# In[14]:


#search_space for hyperparameter tuning

search_space = {"learning_rate" : [0.1, 0.01], 
                "num_hidden1": [1024, 512], 
                "num_hidden2": [128, 64]}


# In[15]:


#this LossHistory class will monitor the model training using callbacks

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


# In[17]:


#apply hyperparameter tuning

tuning_result_dict = {}
for i in range(2):
    # build model
    model = build_model(4096, search_space.get("num_hidden1")[i], search_space.get("num_hidden2")[i])
    history = LossHistory()
    # compile and fit the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=search_space.get("learning_rate")[i]), loss='binary_crossentropy')
    model.fit(X_train_reconstructed, X_train_reconstructed, epochs=50, callbacks=[history])
    tuning_result_dict[f"model{i}"] = min(history.losses)


# In[24]:


#printing tuning result
tuning_result_dict


# In[19]:


#prediction on test data
decoded_img_test = model.predict(X_test_reconstructed)


# In[20]:


#image ploting function

def plot_images(n, data_1, data_2):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(data_1[i].reshape(64,64))
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(data_2[i].reshape(64,64))
        plt.title("new")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show()


# In[33]:


# ploting 5 images with reconstructed and decoded on test data

plot_images(5, X_test_reconstructed, decoded_img_test)


# In[28]:


# prediction on validation data

decoded_img_valid = model.predict(X_valid_reconstructed)


# In[31]:


# ploting 5 images with reconstructed and decoded on validation data

plot_images(5, X_valid_reconstructed, decoded_img_valid)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




