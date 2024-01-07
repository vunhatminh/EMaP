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
from captum.attr import Lime
from custom_captume_lime import LimeBase
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso, SkLearnLinearModel
from captum.attr._core.lime import get_exp_kernel_similarity_function
from captum.metrics._core.infidelity import infidelity

import os
import json

from fashion_model import FashionCNN 
from manifold_torch import Manifold_Image_Sampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def binarize_tensor(tensor, number_of_ones):
    binary_tensor = torch.zeros_like(tensor.reshape(rows*cols))
    _, top_indices = torch.topk(tensor.reshape(rows*cols), number_of_ones, sorted=False)
    binary_tensor[top_indices] = 1

    return binary_tensor.reshape(1,1,rows,cols)

def rdt_fidelity(model, inp, exp, target, samples = 100, num_ones = 10, noise_level = 0.2):
    correct = 0.0
    
    mask = binarize_tensor(exp, num_ones)
    
    for i in range(samples):
        perturb = torch.clip(inp + (1 - mask) * (torch.rand_like(inp)-0.5)*noise_level,0,1)
        log_logits = model(perturb)
        distorted_labels = log_logits.argmax(dim=-1)
        if distorted_labels == target:
            correct += 1
    return correct / samples

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

def perturb_infidelity(inputs):
    noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape)).float().to(device)
    return noise, inputs - noise

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
    parser.add_argument(
        "--fidnoise", dest="fidnoise", type=float, help="Noise for fidelity computation"
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
        lime = False,
        fidnoise = 0.2
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
FID_NOISE = prog_args.fidnoise

print("EXPERIMENT: ", EXPERIMENT)
print("RADIUS: ", RADIUS)
print("DIM: ", DIM)
print("SIM_SIGMA: ", SIM_SIGMA)

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

print("Initializing LIME")

exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=SIM_SIGMA)
lime_zero = Lime(model, 
                interpretable_model=SkLearnLinearRegression(),  # build-in wrapped sklearn Linear Regression
                similarity_func=exp_eucl_distance)

def to_interp_transform(curr_sample, original_inp,
                                      **kwargs):
     return curr_sample.to(device)
    
def similarity_kernel(
    original_input,
    perturbed_input,
    perturbed_interpretable_input,
    **kwargs):
        # kernel_width will be provided to attribute as a kwarg
        kernel_width = kwargs["kernel_width"]
        l2_dist = torch.norm(original_input - perturbed_input)
        return torch.exp(- (l2_dist**2) / (kernel_width**2))

def perturb_func_random_additive(
     original_input,
     **kwargs):
         return torch.abs(original_input.to(device) +  torch.rand_like(original_input.to(device)).to(device))
    
lime_random_add = LimeBase(model.to(device),
                         SkLearnLinearModel("linear_model.Ridge"),
                         similarity_func=similarity_kernel,
                         perturb_func=perturb_func_random_additive,
                         perturb_interpretable_space=False,
                         from_interp_rep_transform=None,
                         to_interp_rep_transform=to_interp_transform)

def perturb_func_random_mul(
     original_input,
     **kwargs):
         return torch.abs(original_input.to(device) *  torch.randn_like(original_input.to(device)).to(device))
    
lime_random_mul = LimeBase(model.to(device),
                         SkLearnLinearModel("linear_model.Ridge"),
                         similarity_func=similarity_kernel,
                         perturb_func=perturb_func_random_mul,
                         perturb_interpretable_space=False,
                         from_interp_rep_transform=None,
                         to_interp_rep_transform=to_interp_transform)

log_odds_lime_zero = []
infidelity_lime_zero = []
fidelity_lime_zero = []

log_odds_lime_add = []
infidelity_lime_add = []
fidelity_lime_add = []

log_odds_lime_mul = []
infidelity_lime_mul = []
fidelity_lime_mul = []

duration = []

num_ones = []

for index_to_explain in range(NO_SAMPLES):
    start_time = time.time()
    image_to_explain = all_images[index_to_explain]

    original_output = model(image_to_explain.unsqueeze(0).to(device))
    original_prob = nn.functional.softmax(original_output, dim = 1)
    first_prediction, second_prediction = torch.topk(original_output, 2)[1][0]

    original_log_odds = np.log(original_prob[0][first_prediction].cpu().detach().numpy()/original_prob[0][second_prediction].cpu().detach().numpy())

    start_time = time.time()
    lime_zero_1st = lime_zero.attribute(
        image_to_explain.unsqueeze(0).to(device),
        target=first_prediction,
        n_samples=1000,
        perturbations_per_eval=16,
        show_progress=False
    )
    duration.append(time.time() - start_time)

    lime_zero_2nd = lime_zero.attribute(
            image_to_explain.unsqueeze(0).to(device),
            target=second_prediction,
            n_samples=1000,
            perturbations_per_eval=16,
            show_progress=False
        )
    
    modified_image_lime = image_to_explain.clone().numpy()
    modified_mask_lime = gen_mask(lime_zero_1st.squeeze(0).cpu().detach().numpy()[0] - lime_zero_2nd.squeeze(0).cpu().detach().numpy()[0])
    modified_image_lime[0][modified_mask_lime==1] = 1 - modified_image_lime[0][modified_mask_lime==1]
    modified_image_lime = torch.tensor(modified_image_lime)
    modified_output_lime = model(modified_image_lime.unsqueeze(0).to(device))
    modified_prob_lime = nn.functional.softmax(modified_output_lime, dim = 1)
    modified_log_odds_lime = np.log(modified_prob_lime[0][first_prediction].cpu().detach().numpy()/modified_prob_lime[0][second_prediction].cpu().detach().numpy())
    log_odds_score_lime = original_log_odds - modified_log_odds_lime
    log_odds_lime_zero.append(log_odds_score_lime)
    
    lime_rand_add_1st = lime_random_add.attribute(
            image_to_explain.unsqueeze(0).to(device),
            target=first_prediction,
            n_samples=1000,
            perturbations_per_eval=16,
            show_progress=False,
            kernel_width=1000
                ).reshape(1,1,rows, cols)
    
    lime_rand_add_2nd = lime_random_add.attribute(
            image_to_explain.unsqueeze(0).to(device),
            target=second_prediction,
            n_samples=1000,
            perturbations_per_eval=16,
            show_progress=False,
            kernel_width=1000
                ).reshape(1,1,rows, cols)
    
    modified_image_lime = image_to_explain.clone().numpy()
    modified_mask_lime = gen_mask(lime_rand_add_1st.squeeze(0).cpu().detach().numpy()[0] - lime_rand_add_2nd.squeeze(0).cpu().detach().numpy()[0])
    modified_image_lime[0][modified_mask_lime==1] = 1 - modified_image_lime[0][modified_mask_lime==1]
    modified_image_lime = torch.tensor(modified_image_lime)
    modified_output_lime = model(modified_image_lime.unsqueeze(0).to(device))
    modified_prob_lime = nn.functional.softmax(modified_output_lime, dim = 1)
    modified_log_odds_lime = np.log(modified_prob_lime[0][first_prediction].cpu().detach().numpy()/modified_prob_lime[0][second_prediction].cpu().detach().numpy())
    log_odds_score_lime = original_log_odds - modified_log_odds_lime
    log_odds_lime_add.append(log_odds_score_lime)
    
    lime_rand_mul_1st = lime_random_mul.attribute(
            image_to_explain.unsqueeze(0).to(device),
            target=first_prediction,
            n_samples=1000,
            perturbations_per_eval=16,
            show_progress=False,
            kernel_width=1000
                ).reshape(1,1,rows, cols)
    
    lime_rand_mul_2nd = lime_random_mul.attribute(
            image_to_explain.unsqueeze(0).to(device),
            target=second_prediction,
            n_samples=1000,
            perturbations_per_eval=16,
            show_progress=False,
            kernel_width=1000
                ).reshape(1,1,rows, cols)
    
    modified_image_lime = image_to_explain.clone().numpy()
    modified_mask_lime = gen_mask(lime_rand_mul_1st.squeeze(0).cpu().detach().numpy()[0] - lime_rand_mul_2nd.squeeze(0).cpu().detach().numpy()[0])
    modified_image_lime[0][modified_mask_lime==1] = 1 - modified_image_lime[0][modified_mask_lime==1]
    modified_image_lime = torch.tensor(modified_image_lime)
    modified_output_lime = model(modified_image_lime.unsqueeze(0).to(device))
    modified_prob_lime = nn.functional.softmax(modified_output_lime, dim = 1)
    modified_log_odds_lime = np.log(modified_prob_lime[0][first_prediction].cpu().detach().numpy()/modified_prob_lime[0][second_prediction].cpu().detach().numpy())
    log_odds_score_lime = original_log_odds - modified_log_odds_lime
    log_odds_lime_mul.append(log_odds_score_lime)
        

    infid = infidelity(model.to(device), perturb_infidelity, image_to_explain.unsqueeze(0).to(device), lime_zero_1st.to(device), target=first_prediction)
    infidelity_lime_zero.append(infid.cpu().detach().numpy()[0])
    
    infid = infidelity(model.to(device), perturb_infidelity, image_to_explain.unsqueeze(0).to(device), lime_rand_add_1st.to(device), target=first_prediction)
    infidelity_lime_add.append(infid.cpu().detach().numpy()[0])
    
    infid = infidelity(model.to(device), perturb_infidelity, image_to_explain.unsqueeze(0).to(device), lime_rand_mul_1st.to(device), target=first_prediction)
    infidelity_lime_mul.append(infid.cpu().detach().numpy()[0])
    
    for num_one in range(10,100):    
        fid_zero = rdt_fidelity(model.to(device), image_to_explain.unsqueeze(0).to(device), lime_zero_1st.to(device),  target=first_prediction, samples = 100, num_ones = num_one, noise_level = FID_NOISE)
        fid_add  = rdt_fidelity(model.to(device), image_to_explain.unsqueeze(0).to(device), lime_rand_add_1st.to(device),  target=first_prediction, samples = 100, num_ones = num_one, noise_level = FID_NOISE)
        fid_mul  = rdt_fidelity(model.to(device), image_to_explain.unsqueeze(0).to(device), lime_rand_mul_1st.to(device),  target=first_prediction, samples = 100, num_ones = num_one, noise_level = FID_NOISE)
        
        num_ones.append(num_one)
        fidelity_lime_zero.append(fid_zero)
        fidelity_lime_add.append(fid_add)
        fidelity_lime_mul.append(fid_mul)
    
df_infid = pd.DataFrame({'LIME zero infidelity': infidelity_lime_zero,
                         'LIME zero logodds': log_odds_lime_zero,
                         'LIME addition infidelity': infidelity_lime_add,
                         'LIME addition logodds': log_odds_lime_add,
                         'LIME multiplication infidelity': infidelity_lime_mul,
                         'LIME multiplication logodds': log_odds_lime_mul,
                       'Time': duration})

df_fid = pd.DataFrame({'LIME zero fidelity': fidelity_lime_zero,
                       'LIME addition fidelity': fidelity_lime_add,
                       'LIME multiplication fidelity': fidelity_lime_mul,
                       'Num features': num_ones})
    
save_file_infid = 'result/explainer_evaluation/LIME_infidelity_and_logodds' + '_' + str(EXPERIMENT) + '_update2023.pickle'
save_file_fid = 'result/explainer_evaluation/LIME_fidelity' + '_' + str(EXPERIMENT) + '_update2023.pickle'


print("Save file to explainer_evaluation")
with open(save_file_infid, 'wb') as output:
    pickle.dump(df_infid, output)
with open(save_file_fid, 'wb') as output:
    pickle.dump(df_fid, output)