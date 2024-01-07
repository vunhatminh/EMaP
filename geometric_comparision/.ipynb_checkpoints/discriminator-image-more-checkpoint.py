import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import time
import pandas as pd
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from numpy import linalg as LA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model

from fashion_model import FashionCNN 
from manifold_torch import Manifold_Image_Sampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normalize(v, r):
    return v/np.sqrt(np.sum(v**2))*r

def get_discriminator(X,y,n_estimators = 100):
    the_rf = RandomForestClassifier(n_estimators=n_estimators).fit(X, y)
    y_pred = the_rf.predict(X)
    the_rf_result = (y_pred == y).sum()
    return the_rf, the_rf_result/y.shape[0]

def get_discriminator_performance(X,y,rf):
    y_pred = rf.predict(X)
    the_rf_result = (y_pred == y).sum()
    return the_rf_result/y.shape[0]

def arg_parse():
    parser = argparse.ArgumentParser(description="Discriminator MNIST")
    parser.add_argument(
            "--exp", dest="exp", help="Experiments"
        )
    parser.add_argument(
            "--radius", dest="radius", type=float, help="Radius of perturbation"
        )
    parser.add_argument(
            "--base", dest="base", type=float, help="Base radius of perturbation"
        )
    parser.add_argument(
        "--multiplier", dest="multiplier", type=int, help="Number of times an image is perturbed"
    )
    parser.add_argument(
        "--dim", dest="dim", type=int, help="Number of low dim"
    )
    parser.add_argument(
        "--runs", dest="runs", type=int, help="Number of runs"
    )
    parser.add_argument(
        "--target", dest="target", type=int, help="Sublabel to approximiate local hyperplane"
    )
    parser.add_argument(
        "--shuffle", dest="shuffle", type=bool, help="Shuffle the pivots"
    )
    
        
    parser.set_defaults(
        exp = 'mnist',
        radius = 0.00001,
        base = 0.0,
        runs = 10,
        multiplier = 100,
        dim = 2,
        shuffle = True,
        target = None
    )
    return parser.parse_args()

prog_args = arg_parse()

EXPERIMENT = prog_args.exp
RADIUS = prog_args.radius
BASE_RADIUS = prog_args.base
MULTIPLIER = prog_args.multiplier
DIM = prog_args.dim
SHUFFLE = prog_args.shuffle
RUNS = prog_args.runs
TARGET = prog_args.target

print("EXPERIMENT: ", EXPERIMENT)
print("TARGET: ", TARGET)
print("RUNS: ", RUNS)
print("RADIUS: ", RADIUS)

if EXPERIMENT == 'fashion_mnist':
    print("Loading fashion mnist")
    train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                    transforms.Compose([transforms.ToTensor()]))
    test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                                   transforms.Compose([transforms.ToTensor()]))
elif EXPERIMENT == 'mnist':
    print("Loading mnist")
    train_set = torchvision.datasets.MNIST("./data", download=True, transform=
                                                    transforms.Compose([transforms.ToTensor()]))
    test_set = torchvision.datasets.MNIST("./data", download=True, train=False, transform=
                                                   transforms.Compose([transforms.ToTensor()]))
else:
    print("Nothing to do.")
    
print("Done loading")
    
train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size=100)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=100)


all_loader = torch.utils.data.DataLoader(train_set, batch_size=train_set.__len__())
all_images, all_labels = next(iter(all_loader))

start_time = time.time()
manifold_sampler = Manifold_Image_Sampler(all_images, dim = DIM, labels = all_labels)
# Get the hyperplanes
if TARGET == None:
    targets = np.array(torch.unique(all_labels))
    target_str = 'all'
else:
    targets = [TARGET]
    target_str = str(TARGET)
_ = manifold_sampler.get_pivots(manifold_sampler.labels, MULTIPLIER, shuffle = SHUFFLE, target_labels=targets)
manifold_G = manifold_sampler.get_G_from_pivots()
Gu, Gd, Gv = np.linalg.svd(manifold_G, full_matrices=False)

duration = time.time() - start_time
print("Initialize duration: ", duration)

if TARGET == None:
    targets = np.array(torch.unique(all_labels))
    target_str = 'all'
else:
    targets = [TARGET]
    target_str = str(TARGET)

tp_lime = []
tn_lime = []
tp_shap = []
tn_shap = []
tp_zero = []
tn_zero = []

for run in range(RUNS):
    p_select = 0.8
    ori_pc = manifold_sampler.to_1d(manifold_sampler.pivots).cpu().detach().numpy()
    mask = np.random.uniform(0, 1, size=ori_pc.shape)
    mask[mask > p_select] = 1
    mask[mask <= p_select] = 0
    
    zero_base = np.zeros(ori_pc.shape)
    
    lime_zero_pc = ori_pc.copy()
    lime_zero_pc[mask != 0] = zero_base[mask != 0]
    
    data_max = manifold_sampler.data_max.cpu().item()
    data_min = manifold_sampler.data_min.cpu().item()
    noise_uniform = np.random.uniform(data_min, data_max, size=ori_pc.shape)
    lime_noise_mask = noise_uniform * mask
    lime_pc = ori_pc.copy()
    lime_pc[lime_noise_mask != 0] = lime_noise_mask[lime_noise_mask != 0]
    
    SHAP_bg = ori_pc.mean(axis = 0)
    SHAP_bg = np.tile(SHAP_bg,(ori_pc.shape[0],1))
    shap_noise_mask = SHAP_bg * mask
    shap_pc = ori_pc.copy()
    shap_pc[shap_noise_mask != 0] = shap_noise_mask[shap_noise_mask != 0]

    X_discriminator_zero = np.vstack((ori_pc[::2], lime_zero_pc[::2]))
    X_discriminator_lime = np.vstack((ori_pc[::2], lime_pc[::2]))
    X_discriminator_shap = np.vstack((ori_pc[::2], shap_pc[::2]))
    
    y_discriminator = np.concatenate((np.zeros(ori_pc[::2].shape[0]), np.ones(ori_pc[::2].shape[0])))
    
    the_rf_zero, train_acc_zero = get_discriminator(X_discriminator_zero, y_discriminator, n_estimators = 100)
    the_rf_lime, train_acc_lime = get_discriminator(X_discriminator_lime, y_discriminator, n_estimators = 100)
    the_rf_shap, train_acc_shap = get_discriminator(X_discriminator_shap, y_discriminator, n_estimators = 100)
    
    tp_lime.append(get_discriminator_performance(lime_pc[1::2],      np.ones(lime_pc[1::2].shape[0]) , the_rf_lime))
    tn_lime.append(get_discriminator_performance(ori_pc[1::2] ,      np.zeros(ori_pc[1::2].shape[0]) , the_rf_lime))
    tp_shap.append(get_discriminator_performance(shap_pc[1::2],      np.ones(shap_pc[1::2].shape[0]) , the_rf_shap))
    tn_shap.append(get_discriminator_performance(ori_pc[1::2] ,      np.zeros(ori_pc[1::2].shape[0]) , the_rf_shap))    
    tp_zero.append(get_discriminator_performance(lime_zero_pc[1::2], np.ones(lime_zero_pc[1::2].shape[0]) , the_rf_zero))
    tn_zero.append(get_discriminator_performance(ori_pc[1::2] ,      np.zeros(ori_pc[1::2].shape[0]) , the_rf_zero))

df = pd.DataFrame({'True positive lime zero': tp_zero,
                   'True negative lime zero': tn_zero,
                   'True positive lime': tp_lime,
                   'True negative lime': tn_lime,
                   'True positive shap': tp_shap,
                   'True negative shap': tn_shap,})

discriminator_file = 'result/discriminator/' + EXPERIMENT + '_label_' + target_str + '_dim_' + str(DIM) + '_std_' + str(RADIUS) +'_zero_base.pickle'

print("Save file to ", discriminator_file)
with open(discriminator_file, 'wb') as output:
    pickle.dump(df, output)