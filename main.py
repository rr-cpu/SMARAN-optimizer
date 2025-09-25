import torch
import torch.nn as nn
import torch.optim as optim
import utils
import copy
from datasets import CIFAR100, CIFAR10, TinyImageNetLoader
from optimizers import DecGD, SMARAN
from models import DenseNet,ResNet50
from trainer import Trainer
from utils import save_losses_to_excel,plot_losses_from_folder,seed_everything
import random
import numpy as np
import os
import config

if __name__ == '__main__':
    
    seed_everything(config.seed)
    # Check for available GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #common hyperparameters
    batch_size=config.batch_size
    epochs=config.epochs    
    model_type=config.model_type
    dataset_name=config.dataset_type   
    #loss initialization
    criterion = nn.CrossEntropyLoss()
    #model and data initialization
    if model_type=='ResNet50' and dataset_name=='CIFAR10':
        model=ResNet50(num_classes=10)
        dataset = CIFAR10(batch_size=batch_size)
    elif model_type=='ResNet50' and dataset_name=='CIFAR100':
        model=ResNet50(num_classes=100)
        dataset = CIFAR100(batch_size=batch_size)
    elif model_type=='DenseNet' and dataset_name=='CIFAR10':
        model=DenseNet(num_classes=10)
        dataset = CIFAR10(batch_size=batch_size)
    elif model_type=='DenseNet' and dataset_name=='CIFAR100':
        model=DenseNet(num_classes=100)
        dataset = CIFAR100(batch_size=batch_size)
    elif model_type=='ResNet50' and dataset_name=='tinyimagenet':
        model=ResNet50(num_classes=200)
        dataset=TinyImageNetLoader("./data/tiny-imagenet/tiny-imagenet-200", batch_size=batch_size)
    elif model_type=='DenseNet' and dataset_name=='tinyimagenet':
        model=DenseNet(num_classes=200)
        dataset=TinyImageNetLoader("./data/tiny-imagenet/tiny-imagenet-200", batch_size=batch_size)
    #trainer initialization 
    trainer=Trainer(device)
    #optimizers list
    optimizer_names=['SMARAN','prodigy','RAdam','SGD','Adam','AdamW','DecGD','SGDM','sps','PoNoS','lion']
    print(model_type,dataset_name)
    #training process
    trainer.train(model, optimizer_names, dataset, criterion, epochs=epochs)
    #plot the results
    plot_losses_from_folder('./results')