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


# In[ ]:


class Network(nn.Module):
  def __init__(self,feature_extractor,input_features_for_denselayer):
    super(Network, self).__init__()
    self.extractor=feature_extractor
    self.RHC=nn.Sequential(
                          nn.Linear(input_features_for_denselayer,4096),
                          nn.ReLU(inplace=True),
                          nn.Dropout(p=0.5,inplace=False),
                          nn.Linear(4096,1024),
                          nn.ReLU(inplace=True),
                          nn.Dropout(p=0.5,inplace=False),
                          nn.Linear(1024,128),
                          nn.ReLU(inplace=True),
                          nn.Dropout(p=0.5,inplace=False),
                          nn.Linear(128,8))
    self.input_features_for_denselayer=input_features_for_denselayer
    
  def forward(self,x):
    x=self.extractor(x)
    x=x.view(-1,self.input_features_for_denselayer)
    x=self.RHC(x)
    return x

class new_model(nn.Module):
  def __init__(self,feature_extractor,input_features_for_denselayer):
    super(new_model, self).__init__()
    self.extractor=feature_extractor
    self.RHC=nn.Sequential(
                          nn.Linear(input_features_for_denselayer,4096),
                          nn.ReLU(inplace=True),
                          nn.Linear(4096,8))
    self.input_features_for_denselayer=input_features_for_denselayer
    
  def forward(self,x):
    x=self.extractor(x)
    x=x.view(-1,self.input_features_for_denselayer)
    x=self.RHC(x)
    return x

class new_Regressor(nn.Module):
  def __init__(self,feature_extractor,input_features_for_denselayer):
    super(new_Regressor, self).__init__()
    self.extractor=feature_extractor
    self.RHC1=nn.Sequential(
                          nn.Linear(input_features_for_denselayer,4096),
                          nn.ReLU(inplace=True),
                          nn.Linear(4096,512),
                          nn.ELU(inplace=True),
                          nn.Linear(512,2))
#     self.RHC2=nn.Sequential(
#                           nn.Linear(input_features_for_denselayer,4096),
#                           nn.ReLU(inplace=True),
#                           nn.Linear(4096,512),
#                           nn.ELU(inplace=True),
#                           nn.Linear(512,1))    
    self.input_features_for_denselayer=input_features_for_denselayer
    
  def forward(self,x):
    x=self.extractor(x)
    x=x.view(-1,self.input_features_for_denselayer)
    val=self.RHC1(x)[0]
    aro=self.RHC1(x)[1]
    return val, aro
