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
from Dataset import MyDataset


# In[2]:


def vgg():
    vgg=torchvision.models.vgg16(pretrained=True)
    features=list(vgg.features)[:30]
    features.append(vgg.avgpool)
    for layer in features[:23]:
        for p in layer.parameters():
            p.requires_grad=False
    return nn.Sequential(*features)


# In[ ]:


def total_parameters(model):
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total += param.numel()
    print()
    print('Total', '\t', total)

