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

# In[47]:
class MyDataset(Dataset):
    def __init__(self, anno_dir,image_dir,transform=None, target_transform=None):
        self.anno_dir = anno_dir
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        anno=[os.path.join(self.anno_dir, f) for f in os.listdir(self.anno_dir) if 'exp' in f]
        images=[os.path.join(self.image_dir , f) for f in os.listdir(self.image_dir )]
        anno.sort(key=lambda f: int(re.sub('\D', '', f)))
        images.sort(key=lambda f: int(re.sub('\D', '', f)))
        y=torch.tensor(int(np.load(anno[index])),dtype=torch.float32)
        x=image.imread(images[index])
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x,y
    
    def __len__(self):
        return len(os.listdir(self.image_dir))

# In[ ]:




