#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io import loadmat
import os
import numpy as np
import tqdm


# In[2]:


hea_files = os.listdir("./Training_WFDB/")
hea_files = ["./Training_WFDB/" + i for i in hea_files if "hea" in i]

mat_files = os.listdir("./Training_WFDB/")
mat_files = ["./Training_WFDB/" + i for i in mat_files if "mat" in i]


# In[5]:


max_length = -np.inf
min_length = np.inf

data_points = {}

for f in tqdm.tqdm(mat_files):
    data = loadmat(f)['val']
    data_points[f] = {}
    with open(f.replace("mat", "hea"), "r") as file:
        extra = file.readlines()
    for idx, line in enumerate(extra):
        extra[idx] = line.strip()
        
        if extra[idx].startswith("#Age"):
            age = extra[idx].split(":")[1].strip()
            try:
                data_points[f]['age'] = int(age)
            except:
                data_points[f]['age'] = np.nan
        
        if extra[idx].startswith("#Sex"):
            sex = extra[idx].split(":")[1].strip()
            data_points[f]['sex'] = sex
        
        if extra[idx].startswith("#Dx"):
            label = extra[idx].split(":")[1].strip()
            data_points[f]['label'] =  label

    if data.shape[1] > max_length:
        max_length = data.shape[1]
    if data.shape[1] < min_length:
        min_length = data.shape[1]
    data = np.pad(data, ((0, 0), (0, 72000 - data.shape[1])), 'constant', constant_values=(0))
    data_points[f]['vals'] = data


# In[6]:


import pickle


# In[ ]:


with open("./CleanData.pkl", "wb") as f:
    pickle.dump(data_points, f)


# In[ ]:




