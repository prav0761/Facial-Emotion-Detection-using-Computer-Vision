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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    for batch,(X,y_exp,y_val,y_aro) in enumerate(tqdm(dataloader)):
        y_val = y_val.type(torch.LongTensor)
        y_aro = y_aro.type(torch.LongTensor)
        X,y_val,y_aro=X.to(device),y_val.to(device),y_aro.to(device)

        predVal, predAro = model(X)
        loss1=loss_fn(pred,predVal)
        loss2=loss_fn(pred,predAro)

        optimizer.zero_grad()
        loss1.backward()
        loss2.backward()
        optimizer.step()
    loss=loss.item()
    print(f'loss:{loss:>5f}',f'batch:{batch}/{len(dataloader)}')  
    
def train_batch(dataloader,model,loss_fn,optimizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    for batch,(X,y_exp,y_val,y_aro) in enumerate(tqdm(dataloader)):
        y_val = y_val.type(torch.LongTensor).view(-1,1)
        y_aro = y_aro.type(torch.LongTensor).view(-1,1)
        X,y_val,y_aro=X.to(device),y_val.to(device),y_aro.to(device)

        predVal, predAro = model(X)
        loss1=loss_fn(predVal,y_val)
        loss2=loss_fn(predAro,y_aro)

        optimizer.zero_grad()
        loss1.backward()
        loss2.backward()
        optimizer.step()
        if batch%2==0:
            loss=loss.item()
            print(f'loss:{loss:>5f}',f'batch:{batch}/{len(dataloader)}') 

def validation(dataloader,model,loss_fn):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss1,test_loss2,correct1, correct2=0,0,0,0
    with torch.no_grad():
        for batch,(X,y_exp,y_val,y_aro) in enumerate(tqdm(dataloader)):
            y_val = y_val.type(torch.LongTensor)
            y_aro = y_aro.type(torch.LongTensor)
            X,y_val,y_aro=X.to(device),y_val.to(device),y_aro.to(device)
            
            predVal, predAro = model(X)
            test_loss1+=loss_fn(predVal,y_val).item()
            test_loss2+=loss_fn(predAro,y_aro).item()
            
            correct1+=(predVal.argmax(1)==y).sum().item()
            correct2+=(predAro.argmax(1)==y).sum().item()
    test_loss1/=num_batches
    correct1/=size
    
    test_loss2/=num_batches
    correct2/=size
    print(f'test error for val-{test_loss1:>5f} \n Accuracy for val-{correct1*100:>3f}%')
    print(f'test error for aro-{test_loss2:>5f} \n Accuracy for aro-{correct2*100:>3f}%')

def validation_classes(dataloader,model,loss_fn,label):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    X_c=0
    test_loss,correct=0,0
    with torch.no_grad():
        for batch,(X,y) in enumerate(tqdm(dataloader)):
            y = y.type(torch.LongTensor)
            label1=torch.tensor(label).type(torch.LongTensor)
            if y!=label1:
                continue
            X,y=X.to(device),y.to(device)
            pred=model(X)
            X_c+=1
            test_loss+=loss_fn(pred,y).item()
            correct+=(pred.argmax(1)==y).sum().item()
    test_loss/=X_c
    correct/=X_c
    print(f'test error-{test_loss:>5f} \n Accuracy-{correct*100:>3f}%')

def image_validation(model,data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
            pred=model(pred_x.unsqueeze(0))
            predicted=pred.argmax(1)
            print('predicted is',classes[predicted])
        count+=1

