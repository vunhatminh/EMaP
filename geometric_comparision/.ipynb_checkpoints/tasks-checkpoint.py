import numpy as np
from numpy import linalg as LA
import cv2
import pickle
import matplotlib.pyplot as plt
import time
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model

from manifold_torch import Manifold_Image_Sampler

def mnist(args):
    print("--- Loading mnist")
    train_set = torchvision.datasets.MNIST("./data", download=True, transform=
                                                    transforms.Compose([transforms.ToTensor()]))
    test_set = torchvision.datasets.MNIST("./data", download=True, train=False, transform=
                                                   transforms.Compose([transforms.ToTensor()]))
    
    train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size=100)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=100)


    all_loader = torch.utils.data.DataLoader(train_set, batch_size=train_set.__len__())
    all_images, all_labels = next(iter(all_loader))
    
def fashion(args):
    print(args)