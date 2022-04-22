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



# In[2]:


def full_vgg():
        vgg=torchvision.models.vgg16(pretrained=True)
        return vgg



def vgg_all_freezed():
    vgg=torchvision.models.vgg16(pretrained=True)
    features=list(vgg.features)[:30]
    features.append(vgg.avgpool)
    for layer in features[:30]:
        for p in layer.parameters():
            p.requires_grad=False
    return nn.Sequential(*features)


def vgg_layer_freeze(index):
    vgg=torchvision.models.vgg16(pretrained=True)
    features=list(vgg.features)[:30]
    features.append(vgg.avgpool)
    for layer in features[:index]:
        for p in layer.parameters():
            p.requires_grad=False
    return nn.Sequential(*features)


# In[ ]:


def total_trainable_parameters(model):
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total += param.numel()
    print()
    print(f'Total trainable parameters of {model.__class__.__name__} is', '\t', total)

# In[ ]:




