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

from ripser import Rips
from persim import plot_diagrams, bottleneck

from manifold_torch import Manifold_Image_Sampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def normalize(v, r):
    return v/np.sqrt(np.sum(v**2))*r

def arg_parse():
    parser = argparse.ArgumentParser(description="Persistence diagram MNIST")
    parser.add_argument(
            "--exp", dest="exp", help="Experiments"
        )
    parser.add_argument(
            "--radius", dest="radius", type=float, help="Radius of perturbation"
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
        radius = 0.001,
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
MULTIPLIER = prog_args.multiplier
DIM = prog_args.dim
SHUFFLE = prog_args.shuffle
RUNS = prog_args.runs
TARGET = prog_args.target

print("EXPERIMENT: ", EXPERIMENT)
print("TARGET: ", TARGET)
print("RUNS: ", RUNS)
print("RADIUS: ", RADIUS)
print("MULTIPLIER: ", MULTIPLIER)
print("DIM: ", DIM)

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
manifold_sampler = Manifold_Image_Sampler(all_images, dim = DIM, labels = all_labels)
duration = time.time() - start_time
print("Initialize duration: ", duration)

if TARGET == None:
    targets = np.array(torch.unique(all_labels))
    target_str = 'all'
else:
    targets = [TARGET]
    target_str = str(TARGET)

# Get the hyperplanes
_ = manifold_sampler.get_pivots(manifold_sampler.labels, MULTIPLIER, shuffle = SHUFFLE, target_labels=targets)
manifold_G = manifold_sampler.get_G_from_pivots()
Gu, Gd, Gv = np.linalg.svd(manifold_G, full_matrices=False)

# Computing bottleneck distances

H0s_gauss = []
H0s_plane = []
H0s_ortho = []
H1s_gauss = []
H1s_plane = []
H1s_ortho = []

for run in range(RUNS):
    start_time = time.time()
    gauss_ = np.random.normal(0, 1, size=manifold_sampler.pivots.shape)
    gauss_noise = manifold_sampler.to_1d(gauss_)
    plane_noise = np.zeros_like(gauss_noise)

    for d in range(Gv.shape[0]):
        proj = np.dot(gauss_noise, Gv[d])
        for s in range(plane_noise.shape[0]):
            plane_noise[s] = plane_noise[s] + proj[s]*Gv[d]        
    ortho_noise = gauss_noise - plane_noise

    # noise
    ortho_norm = normalize(ortho_noise, RADIUS)
    plane_norm = normalize(plane_noise, RADIUS)
    gauss_norm = normalize(gauss_noise, RADIUS)
    
    # point clouds
    ortho_pc = manifold_sampler.to_1d(manifold_sampler.pivots) + ortho_norm
    plane_pc = manifold_sampler.to_1d(manifold_sampler.pivots) + plane_norm
    gauss_pc = manifold_sampler.to_1d(manifold_sampler.pivots) + gauss_norm
    
    ori_pc = manifold_sampler.to_1d(manifold_sampler.pivots).cpu().detach().numpy()
    ortho_pc = ortho_pc.cpu().detach().numpy()
    plane_pc = plane_pc.cpu().detach().numpy()
    gauss_pc = gauss_pc.cpu().detach().numpy()

    # diagrams
    diagrams_ori= Rips().fit_transform(ori_pc/RADIUS)
    diagrams_gauss = Rips().fit_transform(gauss_pc/RADIUS)
    diagrams_plane = Rips().fit_transform(plane_pc/RADIUS)
    diagrams_ortho = Rips().fit_transform(ortho_pc/RADIUS)

    # Record H0
    H0_gauss = bottleneck(diagrams_ori[0], diagrams_gauss[0])
    H0_plane = bottleneck(diagrams_ori[0], diagrams_plane[0])
    H0_ortho = bottleneck(diagrams_ori[0], diagrams_ortho[0])
    
    # Record H1
    H1_gauss = bottleneck(diagrams_ori[1], diagrams_gauss[1])
    H1_plane = bottleneck(diagrams_ori[1], diagrams_plane[1])
    H1_ortho = bottleneck(diagrams_ori[1], diagrams_ortho[1])
    
    H0s_gauss.append(H0_gauss)
    H0s_plane.append(H0_plane)
    H0s_ortho.append(H0_ortho)
    H1s_gauss.append(H1_gauss)
    H1s_plane.append(H1_plane)
    H1s_ortho.append(H1_ortho)
    
    duration = time.time() - start_time
    print("Run duration: ", duration)

df = pd.DataFrame({'H0_gauss': H0s_gauss,
                   'H1_gauss': H1s_gauss,
                   'H0_plane': H0s_plane,
                   'H1_plane': H1s_plane,
                   'H0_ortho': H0s_ortho,
                   'H1_ortho': H1s_ortho})

bottleneck_file = 'result/bottleneck/' + EXPERIMENT + '_label_' + target_str + '_dim_' + str(DIM) + '_std_' + str(RADIUS) +'_.pickle'
print("Save file to ", bottleneck_file)
with open(bottleneck_file, 'wb') as output:
    pickle.dump(df, output)
