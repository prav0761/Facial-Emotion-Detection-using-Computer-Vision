#!/usr/bin/env python
# coding: utf-8

# In[18]:


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


# In[21]:


def train(dataloader,model,loss_fn,optimizer):
    model.train()
    for batch,(X,y) in enumerate(tqdm(dataloader)):
        y = y.type(torch.LongTensor)
        X,y=X.to(device),y.to(device)

        pred=model(X)
        loss=loss_fn(pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss=loss.item()
    print(f'loss:{loss:>5f}',f'batch:{batch}/{len(dataloader)}')  


# In[22]:


def validation(dataloader,model,loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss,correct=0,0
    with torch.no_grad():
        for batch,(X,y) in enumerate(tqdm(dataloader)):
            y = y.type(torch.LongTensor)
            X,y=X.to(device),y.to(device)
            pred=model(X)
            test_loss+=loss_fn(pred,y).item()
            correct+=(pred.argmax(1)==y).sum().item()
    test_loss/=num_batches
    correct/=size
    print(f'test error-{test_loss:>5f} \n Accuracy-{correct*100:>3f}%')


# In[ ]:


def image_validation(model,data);
    model.eval()
    classes=['Neutral','Happy','Sad','Surprise','Fear','Disgust','Anger','Contempt']
    test_loader=torch.utils.data.DataLoader(data,batch_size=None,shuffle=False)
    count=0
    for pred_x,pred_y in test_loader:
        if count>20:
        
            break
        plt.imshow(pred_x.permute(1,2,0).squeeze(2))
        plt.show()
        pred_x,pred_y=pred_x.to(device),(torch.tensor(pred_y)).to(device)
        print('real value is ',classes[int(pred_y)])
        with torch.no_grad():
            pred=model1(pred_x.unsqueeze(0))
            predicted=pred.argmax(1)
            print('predicted is',classes[predicted])
        count+=1

