#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from prepare_dataset import MyDataset


# In[3]:


def display_some_images(dataloader):
    classes=['Neutral','Happy','Sad','Surprise','Fear','Disgust','Anger','Contempt']
    plt.figure(figsize=(10,7))
    for i in range(10):
        plt.subplot(2,5,i+1)
        image,label=dataloader[i]
        plt.title(classes[int(label)])
        plt.imshow(image.permute(1,2,0))
    plt.show()
    
def display_image(image,label):
    classes=['Neutral','Happy','Sad','Surprise','Fear','Disgust','Anger','Contempt']
    plt.title(classes[int(label)])
    plt.imshow(image.permute(1,2,0))
    plt.show()

def display_image_transform_row(image,t_image):
    plt.figure()

    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1,4,figsize=(15,15)) 

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    axarr[0].imshow(image.permute(1,2,0))
    axarr[0].set_title("Original")
    axarr[1].imshow(t_image.tran(image).permute(1,2,0))
    axarr[1].set_title("Transform 1")
    axarr[2].imshow(t_image.tran(image).permute(1,2,0))
    axarr[2].set_title("Transform 2")
    axarr[3].imshow(t_image.tran(image).permute(1,2,0))
    axarr[3].set_title("Transform 3")
    for ax in axarr:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
def stats_of_image_and_label(image,label):
    print(type(image))
    print(type(label))
    print(image.shape)
    print(label.shape)
    print(image.dtype)
    print(label.dtype)

def MyCuda_Stats():
    print('No of GPUs i have is',torch.cuda.device_count())
    print(torch.cuda.current_device())
    print('My Graphic Card is',torch.cuda.get_device_name(torch.cuda.current_device()))
    print('Is Cuda Available',torch.cuda.is_available())

def get_output_features_of_model(model,batch_size,dataloader):
    for a,b in dataloader:
        a,b=a,b
        break
    return (model(a)).view(batch_size,-1).shape[1]



