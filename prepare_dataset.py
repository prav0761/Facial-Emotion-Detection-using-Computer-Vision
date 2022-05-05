#!/usr/bin/env python
# coding: utf-8

# In[32]:


import torch
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
from tqdm import tqdm

# In[47]:
class MyDataset(Dataset):
    def __init__(self, anno_dir,image_dir,transform=None, target_transform=None):
        self.image_dir_for_len_purpose = image_dir
        self.anno_dir1=[os.path.join(anno_dir, f) for f in os.listdir(anno_dir) if 'exp' in f]
        self.images_dir1=[os.path.join(image_dir , f) for f in os.listdir(image_dir )]
        self.anno_dir1.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.images_dir1.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        anno_dir2=self.anno_dir1
        images_dir2=self.images_dir1
        y=torch.tensor(int(np.load(anno_dir2[index])),dtype=torch.float32)
        x=image.imread(images_dir2[index])
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x,y
    
    def __len__(self):
        return len(self.anno_dir1)
    
    def upsample(self):
        classes=['Neutral','Happy','Sad','Surprise','Fear','Disgust','Anger','Contempt']
        count_dict={}
        idx_dict={}
        for c in classes:
            count_dict[c]=0
            idx_dict[c]=[]
        print("Finding Class Counts...")
        for idx in tqdm(range(len(self.anno_dir1))):
            label = torch.tensor(int(np.load(self.anno_dir1[idx])),dtype=torch.float32)
            count_dict[classes[int(label)]]+=1
            idx_dict[classes[int(label)]].append(idx)
        class_max = 20000
        #for c in classes:
            #if count_dict[c]>class_max:
                #class_max = count_dict[c]
        for c in classes:
            print("Balancing Class:",c)
            add_sampl = class_max-count_dict[c]
            add_sampl = np.max([add_sampl,0])
            samples_idx = np.random.choice(idx_dict[c], add_sampl, replace=True)
            for s in tqdm(samples_idx):
                assert(classes[int(torch.tensor(int(np.load(self.anno_dir1[s])),dtype=torch.float32))]==c)
                self.images_dir1.append(self.images_dir1[s])
                self.anno_dir1.append(self.anno_dir1[s])

def subset_generator(data,count):
    indices = torch.randperm(len(data))[:count]
    Subset_sampler=torch.utils.data.SubsetRandomSampler(indices, generator=None)
    return Subset_sampler




