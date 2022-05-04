#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import re
import os
from PIL import Image
from torchvision.io import read_image
from matplotlib import image
from prepare_dataset import MyDataset,subset_generator
from utils import display_some_images,stats_of_image_and_label,MyCuda_Stats,get_output_features_of_model
from vgg_pretrained import full_vgg,vgg_all_freezed,vgg_layer_freeze,total_trainable_parameters
from tqdm import tqdm
import torchvision.transforms as T
import random



# In[ ]:

#T.RandomRotation(degrees=(0, 180))
#T.RandomHorizontalFlip(p=0.5)
class Transform():
  def __init__(self,input_features):
    self.transform = T.Compose([T.RandomHorizontalFlip(p=0.5),
                          ])
    
    #self.t_image=nn.Sequential(transform(input_features,input_features),)
    self.input_features=input_features
    
  def tran(self,x):
    x=self.transform(x)
    x=T.functional.rotate(img=x,angle =random.choice([0,90,180,270]))
    x=self.transform(x)
    return x


