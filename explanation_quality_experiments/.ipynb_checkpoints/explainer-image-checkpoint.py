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
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from numpy import linalg as LA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model

from captum.attr import visualization as viz
from captum.attr import Lime, LimeBase
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function

import os
import json

from fashion_model import FashionCNN 
from manifold_torch import Manifold_Image_Sampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def similarity_kernel(v1,v2,kernel_width = 1):
    l2_dist = np.linalg.norm(v1 - v2)
    return np.exp(- (l2_dist**2) / (kernel_width**2))

def normalize(v, r):
    return v/np.sqrt(np.sum(v**2))*r

def gen_mask(score, ones_ratio = 0.2):
    no_rows, no_cols = score.shape
    score_flat = np.reshape(score, no_rows*no_cols)
    top_k = int(ones_ratio*no_rows*no_cols)
    idx = np.argpartition(score_flat, -top_k)[-top_k:]
    indices = idx[np.argsort((-score_flat)[idx])]
    score_flat_dup = np.zeros_like(score_flat)
    score_flat_dup[indices] = 1.0
    score_dup = score_flat_dup.reshape(no_rows, no_cols)
    return score_dup

def arg_parse():
    parser = argparse.ArgumentParser(description="Explanation Evaluation MNIST")
    parser.add_argument(
            "--exp", dest="exp", help="Experiments"
        )
    parser.add_argument(
            "--no_samples", dest="no_samples", type=int, help="Number of samples for evaluation"
        )
    parser.add_argument(
            "--radius", dest="radius", type=float, help="Radius of perturbation"
        )
    parser.add_argument(
            "--sigma", dest="sigma", type=float, help="Similarity kernel width"
        )
    parser.add_argument(
        "--multiplier", dest="multiplier", type=int, help="Number of samples in each class in training the sampler"
    )
    parser.add_argument(
        "--no_perturbations", dest="no_perturbations", type=int, help="Number of times the perturbations are duplicated"
    )
    parser.add_argument(
        "--dim", dest="dim", type=int, help="Number of low dim"
    )
    parser.add_argument(
        "--shuffle", dest="shuffle", type=bool, help="Shuffle the pivots"
    )
    parser.add_argument(
        "--lime", dest="lime", type=bool, help="Compare with LIME"
    )
    
        
    parser.set_defaults(
        exp = 'mnist',
        no_samples = 1,
        radius = 0.00001,
        sigma = 1.0,
        runs = 10,
        multiplier = 100,
        no_perturbations = 1,
        dim = 2,
        shuffle = True,
        lime = False
    )
    return parser.parse_args()

prog_args = arg_parse()

EXPERIMENT = prog_args.exp
NO_SAMPLES = prog_args.no_samples
RADIUS = prog_args.radius
NUM_PERTURBATIONS = prog_args.no_perturbations
MULTIPLIER = prog_args.multiplier
DIM = prog_args.dim
SHUFFLE = prog_args.shuffle
SIM_SIGMA = prog_args.sigma
run_LIME = prog_args.lime

print("EXPERIMENT: ", EXPERIMENT)
print("RADIUS: ", RADIUS)
print("DIM: ", DIM)
print("SIM_SIGMA: ", SIM_SIGMA)
print("RUN LIME: ", run_LIME)

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
no_samples, channels, rows, cols = all_images.shape

start_time = time.time()
manifold_sampler = Manifold_Image_Sampler(all_images, dim = DIM, labels = all_labels)
duration = time.time() - start_time
print("Initialize duration: ", duration)

# Get the pivots
_ = manifold_sampler.get_pivots(manifold_sampler.labels, MULTIPLIER, shuffle = SHUFFLE, target_labels=None)

# Get hyperplanes
Gvs = []
for group in torch.unique(manifold_sampler.labels):
    manifold_G = manifold_sampler.get_G_from_samples(manifold_sampler.pivots[group.item()*MULTIPLIER:(group.item()+1)*MULTIPLIER])
    Gu, Gd, Gv = np.linalg.svd(manifold_G, full_matrices=False)
    Gvs.append(Gv)

sample_to_Gv = {}
group = -1
for i in range(manifold_sampler.pivots.shape[0]):
    if i % MULTIPLIER == 0:
        group = group + 1
    sample_to_Gv[i] = group 
print("Done getting hyperplanes")
    
# Load model

if EXPERIMENT == 'fashion_mnist':
    checkpt_file = 'pretrained/fashionCNN.pt'

elif EXPERIMENT == 'mnist':
    checkpt_file = 'pretrained/mnistCNN.pt'

else:
    print("Nothing to do.")

print(checkpt_file)

model = FashionCNN()
model.to(device)
model.load_state_dict(torch.load(checkpt_file))
model.eval()
print("Done loading model")
    
# Get perturbations
BASE_RADIUS = 0.00001
NUM_PERTURBATIONS = 10
perturbations = []
for _ in range(NUM_PERTURBATIONS):
    for group in torch.unique(manifold_sampler.labels):
        base_batch = manifold_sampler.pivots[group.item()*MULTIPLIER:(group.item()+1)*MULTIPLIER]
        
        # base
        base_gauss_ = np.random.normal(0, 1, size=base_batch.shape)
        r = np.random.uniform()*BASE_RADIUS
        base_gauss_norm = normalize(base_gauss_, r)
        base_pc = manifold_sampler.to_1d(base_batch + base_gauss_norm)
    
        # gauss
        gauss_ = np.random.normal(0, 1, size=base_batch.shape)
        gauss_noise = manifold_sampler.to_1d(gauss_)
        plane_noise = np.zeros_like(gauss_noise)
        for d in range(Gvs[group].shape[0]):
            proj = np.dot(gauss_noise, Gvs[group][d])
            for s in range(plane_noise.shape[0]):
                plane_noise[s] = plane_noise[s] + proj[s]*Gvs[group][d]        
        ortho_noise = gauss_noise - plane_noise
    
        # noise
        r = np.random.uniform()*RADIUS
        ortho_norm = normalize(ortho_noise, r)
        ortho_pc = base_pc + ortho_norm
    
        perturbations.append(manifold_sampler.to_3d(ortho_pc))
    
perturbations = torch.cat(perturbations)    
perturb_embeded = manifold_sampler.transform(perturbations)
print("Done generating perturbations")

# Explain 
if NO_SAMPLES > manifold_sampler.pivots.shape[0]:
    NO_SAMPLES = manifold_sampler.pivots.shape[0]

sampler_score = []
sampler_duration = []
if run_LIME == True:
    print("Initializing LIME")
    lime_score = []
    lime_duration = []
    exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=SIM_SIGMA)
    lr_lime = Lime(model, 
                    interpretable_model=SkLearnLinearRegression(),  # build-in wrapped sklearn Linear Regression
                    similarity_func=exp_eucl_distance)

for index_to_explain in range(NO_SAMPLES):
    start_time = time.time()
    image_to_explain = manifold_sampler.pivots[index_to_explain]

    perturb_outputs = model(perturbations.float().to(device))
    probs = nn.functional.softmax(perturb_outputs, dim = 1)

    original_output = model(image_to_explain.unsqueeze(0).to(device))
    original_prob = nn.functional.softmax(original_output, dim = 1)
    first_prediction, second_prediction = torch.topk(original_output, 2)[1][0]
    perturb_1st = probs[:,first_prediction.item()].cpu().detach().numpy()
    perturb_2nd = probs[:,second_prediction.item()].cpu().detach().numpy()

    base_embeded = manifold_sampler.transform(image_to_explain.unsqueeze(0))
    similarities = [similarity_kernel(perturb_embeded[i], base_embeded, kernel_width = SIM_SIGMA) for i in range(perturbations.shape[0])]
    repeat_shape = (perturbations.shape[0],) + tuple(np.ones(image_to_explain.ndim).astype(int))
    repeat_image_to_explain = image_to_explain.repeat(repeat_shape)
    true_perturb = perturbations - repeat_image_to_explain
    clf = linear_model.Ridge(alpha = 200)
    clf.fit(np.abs(true_perturb).reshape(true_perturb.shape[0], channels*rows*cols), perturb_1st, sample_weight=similarities)
    explanation_1st = -clf.coef_.reshape(rows, cols)
    clf.fit(np.abs(true_perturb).reshape(true_perturb.shape[0], channels*rows*cols), perturb_2nd, sample_weight=similarities)
    explanation_2nd = -clf.coef_.reshape(rows, cols)
    
    modified_image = image_to_explain.clone().numpy()
    modified_mask = gen_mask(explanation_1st - explanation_2nd)
    modified_image[0][modified_mask==1] = 1 - modified_image[0][modified_mask==1]
    modified_image = torch.tensor(modified_image)
    modified_output = model(modified_image.unsqueeze(0).to(device))
    modified_prob = nn.functional.softmax(modified_output, dim = 1)

    original_log_odds = np.log(original_prob[0][first_prediction].cpu().detach().numpy()/original_prob[0][second_prediction].cpu().detach().numpy())
    modified_log_odds = np.log(modified_prob[0][first_prediction].cpu().detach().numpy()/modified_prob[0][second_prediction].cpu().detach().numpy())
    log_odds_score = original_log_odds - modified_log_odds
    sampler_score.append(log_odds_score)
    sampler_duration.append(time.time() - start_time)
    
    if run_LIME == True:
        start_time = time.time()
        lime_1st = lr_lime.attribute(
            image_to_explain.unsqueeze(0).to(device),
            target=first_prediction,
            n_samples=200,
            perturbations_per_eval=16,
            show_progress=False
        ).squeeze(0).cpu().detach().numpy()[0]

        lime_2nd = lr_lime.attribute(
            image_to_explain.unsqueeze(0).to(device),
            target=second_prediction,
            n_samples=200,
            perturbations_per_eval=16,
            show_progress=False
        ).squeeze(0).cpu().detach().numpy()[0]
        
        modified_image_lime = image_to_explain.clone().numpy()
        modified_mask_lime = gen_mask(lime_1st - lime_2nd)
        modified_image_lime[0][modified_mask_lime==1] = 1 - modified_image_lime[0][modified_mask_lime==1]
        modified_image_lime = torch.tensor(modified_image_lime)
        modified_output_lime = model(modified_image_lime.unsqueeze(0).to(device))
        modified_prob_lime = nn.functional.softmax(modified_output_lime, dim = 1)

        modified_log_odds_lime = np.log(modified_prob_lime[0][first_prediction].cpu().detach().numpy()/modified_prob_lime[0][second_prediction].cpu().detach().numpy())
        log_odds_score_lime = original_log_odds - modified_log_odds_lime
        lime_score.append(log_odds_score_lime)
        lime_duration.append(time.time() - start_time)

if run_LIME == True:        
    df = pd.DataFrame({'Ortho': sampler_score,
                       'OrthoTime': sampler_duration,
                       'LIME': lime_score,
                       'LIMETime' :lime_duration})
else:
    df = pd.DataFrame({'Ortho': sampler_score, 
                       'OrthoTime': sampler_duration})
    
log_odds_file = 'result/logodd/' + EXPERIMENT + '_dim_' + str(DIM) + '_std_' + str(RADIUS) + '_sigma_' + str(SIM_SIGMA) +'_.pickle'

print("Save file to ", log_odds_file)
with open(log_odds_file, 'wb') as output:
    pickle.dump(df, output)