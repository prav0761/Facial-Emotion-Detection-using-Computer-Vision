#!/usr/bin/env python
# coding: utf-8

# In[32]:


import torch
import math
from time import time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from zipfile import ZipFile
import numpy as np
from time import time
from torchvision import datasets
from torchvision import transforms
import pandas as pd
import numpy as np
import zipfile
import os
from PIL import Image
from torchvision.io import read_image
from matplotlib import image
import re
from sklearn.preprocessing import OneHotEncoder

# In[47]:
class MyDataset(Dataset):
    def __init__(self, anno_dir,image_dir,transform=None, target_transform=None):
        self.image_dir_for_len_purpose = image_dir
        # getting from directory
        self.exp_dir = [os.path.join(anno_dir, f) for f in os.listdir(anno_dir) if 'exp' in f]
        self.val_dir = [os.path.join(anno_dir, f) for f in os.listdir(anno_dir) if 'val' in f]
        self.aro_dir = [os.path.join(anno_dir, f) for f in os.listdir(anno_dir) if 'aro' in f]      
        self.images_dir=[os.path.join(image_dir , f) for f in os.listdir(image_dir )]
        
        #sorting???
        self.exp_dir.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.exp_dir.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.exp_dir.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.images_dir.sort(key=lambda f: int(re.sub('\D', '', f)))
        
        self.transform = transform
        self.target_transform = target_transform
        
        self.df = pd.DataFrame(columns=['val','aro','r','theta','class'])
        for i in range(self.__len__()):
            val = float(np.load(self.val_dir[i]).astype(np.float64,copy=False))
            aro = float(np.load(self.aro_dir[i]).astype(np.float64,copy=False))
            self.df.loc[-1] = [val, aro, math.hypot(val, aro), math.atan2(val,aro), float(np.load(self.exp_dir[i]).astype(np.float64,copy=False))]
            self.df.index = self.df.index + 1
            self.df = self.df.sort_index()
        print("This dataset is done\n")
        return
        
    def __getitem__(self, index):
        y_exp=torch.tensor(int(np.load(self.exp_dir[index])),dtype=torch.float32)
        y_val=torch.tensor(float(np.load(self.val_dir[index])),dtype=torch.float32)
        y_aro=torch.tensor(float(np.load(self.aro_dir[index])),dtype=torch.float32)
        x=image.imread(self.images_dir[index])
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y_exp = self.target_transform(y_exp)
        return x,y_exp,y_val,y_aro       
    
    def basis_to_exp(self):
        return self.df
    
    def __len__(self):
        return len(os.listdir(self.image_dir_for_len_purpose))

def subset_generator(data,count):
    indices = torch.randperm(len(data))[:count]
    Subset_sampler=torch.utils.data.SubsetRandomSampler(indices, generator=None)
    return Subset_sampler




