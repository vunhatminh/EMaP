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
from umap_lime import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def arg_parse():
    parser = argparse.ArgumentParser(description="UMAP discriminator")
    parser.add_argument(
            "--exp", dest="exp", help="Experiments"
        )
    parser.add_argument(
            "--std", dest="std", type=float, help="Perturbation std"
        )
    parser.add_argument(
        "--n", dest="num_images", type=int, help="Number of training images"
    )
    parser.add_argument(
        "--runs", dest="runs", type=int, help="Number of runs"
    )
    parser.add_argument(
        "--multiplier", dest="multiplier", type=int, help="Number of times an image is perturbed"
    )
    parser.add_argument(
        "--dim", dest="dim", type=int, help="Number of low dim"
    )
        
    parser.set_defaults(
        exp = 'fashion_mnist',
        std = 0.1,
        num_images = 100,
        runs = 10,
        multiplier = 1,
        dim = 2
    )
    return parser.parse_args()

prog_args = arg_parse()


EXPERIMENT = prog_args.exp
PERTURBATION_STD = prog_args.std
NUM_IMAGES = prog_args.num_images
NUM_RUNS = prog_args.runs
MULTIPLIER = prog_args.multiplier
DIM = prog_args.dim

# EXPERIMENT = 'fashion_mnist'
# EXPERIMENT = 'mnist'
# EXPERIMENT = 'compass'
# EXPERIMENT = 'german'

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
umap_sampling = UMAP_sampling(all_images, dim = DIM)
umap_duration = time.time() - start_time
print("UMAP duration: ", umap_duration)

def gen_perturbation_train(X, perturbation_multiplier=10, 
                           perturbation_std = 0.3, 
                           method = 'base', 
                           embed_object = None):
    
    all_x, all_x_base, all_y = [], [], []
    
    if method == 'base':
        var = 0
        for _ in range(perturbation_multiplier):
            perturbed_xtrain = np.random.normal(0, perturbation_std, size=X.shape) 
            p_train_x = np.vstack((X, X + perturbed_xtrain))
            p_train_y = np.concatenate((np.ones(X.shape[0]), np.zeros(X.shape[0])))

            all_x.append(p_train_x)
            all_y.append(p_train_y)
            var = var + np.var(perturbed_xtrain)
        all_x = np.vstack(all_x)
        all_y = np.concatenate(all_y)
        
        return all_x, np.sqrt(var/perturbation_multiplier), all_y
    
    elif method == 'umap_ideal':
        assert embed_object != None , 'Must have embedding object (UMAP?)'
        
        var_x = 0
        var_x_base = 0
        
        for _ in range(perturbation_multiplier):
            X_low = embed_object.transform(X) # low(x)
            perturbed_low = np.random.normal(0, perturbation_std, size=X_low.shape)
            X_per_low = X_low + perturbed_low
            X_per_high = embed_object.inv_transform(X_per_low)
            
            std_high = np.sqrt(np.var(X_per_high.numpy() - X.numpy()))
            noise_high = np.random.normal(0, std_high, size=X.shape)
            
            X_per_noise = X + noise_high
            
            p_train_x = np.vstack((X, X_per_high))
            p_noise_x = np.vstack((X, X_per_noise))
            
            var_x = var_x + np.var(X_per_high.numpy() - X.numpy())
            var_x_base = var_x_base + np.var(noise_high)
            
            p_train_y = np.concatenate((np.ones(X.shape[0]), np.zeros(X.shape[0])))

            all_x.append(p_train_x)
            all_x_base.append(p_noise_x)
            all_y.append(p_train_y)
            
        all_x = np.vstack(all_x)
        all_x_base = np.vstack(all_x_base)
        all_y = np.concatenate(all_y)
        
        return all_x, np.sqrt(var_x/perturbation_multiplier), all_x_base, np.sqrt(var_x_base/perturbation_multiplier), all_y
    
    elif method == 'umap_tangent':
        embed_object
        
        
    else:
        return None
    
def get_discriminator_performance(X,y,n_estimators = 100, test_ratio = 0.5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio)
    the_rf = RandomForestClassifier(n_estimators=n_estimators).fit(X_train, y_train)
    y_pred = the_rf.predict(X_test)
    the_rf_result = (y_pred == y_test).sum()
    return the_rf_result/y_test.shape[0], X_train.shape[0]

def get_1d(X):
    n,c,w,h = X.shape
    return X.reshape(n,c*w*h)


print("Number of images for perturbations: ", NUM_IMAGES)
    
    
all_x, std_x, all_x_base, std_base, all_y = gen_perturbation_train(all_images[0:NUM_IMAGES], 
                                                                   perturbation_multiplier=MULTIPLIER, 
                                                                   perturbation_std = PERTURBATION_STD, 
                                                                   method = 'umap_ideal', 
                                                                   embed_object = umap_sampling)

print("***Checking std***")
print(std_x)
print(std_base)

result = []
for test_ratio in list(np.arange(0.5,0.99,0.05)):
    accs_umap = []
    accs_base = []
    for _ in range(NUM_RUNS):
        acc_umap, _ = get_discriminator_performance(get_1d(all_x),all_y, test_ratio = test_ratio)
        acc_base, n = get_discriminator_performance(get_1d(all_x_base),all_y, test_ratio = test_ratio)
        accs_umap.append(acc_umap)
        accs_base.append(acc_base)
    mean_umap = np.mean(np.asarray(accs_umap))
    std_umap = np.std(np.asarray(accs_umap))
    mean_base = np.mean(np.asarray(accs_base))
    std_base = np.std(np.asarray(accs_base))
    result.append((n, mean_base, std_base, mean_umap, std_umap))
    
df = pd.DataFrame.from_records(result, columns =['NoTrain', 'Base', 'std_base', 'Manifold', 'std_manifold'])

discriminator_file = 'results/discriminator/accuracy_on_' + EXPERIMENT + '_dim_' + str(DIM) + '_noise_' + str(PERTURBATION_STD) +'_.pickle'
print("Save file to ", discriminator_file)
with open(discriminator_file, 'wb') as output:
    pickle.dump(df, output)