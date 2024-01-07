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
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
from captum.metrics._core.infidelity import infidelity

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import KernelShap
from captum.attr import DeepLift

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

def rdt_fidelity(model, inp, exp, target, samples = 100, num_ones = 10):
    correct = 0.0
    
    mask = binarize_tensor(exp, num_ones)
    
    for i in range(samples):
        perturb = torch.clip(inp + (1 - mask) * (torch.rand_like(inp)-0.5)*1.0,0,1)
#         perturb = mask * inp + (1 - mask) * torch.rand_like(inp)*0.2
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
        "--shuffle", dest="shuffle", type=bool, help="Shuffle the pivots"
    )
    
        
    parser.set_defaults(
        exp = 'mnist',
        no_samples = 1,
        shuffle = True
    )
    return parser.parse_args()

prog_args = arg_parse()

EXPERIMENT = prog_args.exp
NO_SAMPLES = prog_args.no_samples
SHUFFLE = prog_args.shuffle


print("EXPERIMENT: ", EXPERIMENT)

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

print("Initializing explainers")
shap_ex = KernelShap(model)
deep_ex = DeepLift(model)
gras_ex = GradientShap(model)

logodds_shap = []
infidelity_shap = []
fidelity_shap = []

logodds_deep = []
infidelity_deep = []
fidelity_deep = []

logodds_gras = []
infidelity_gras = []
fidelity_gras = []

duration_shap = []
duration_deep = []
duration_gras = []

num_ones = []

for index_to_explain in range(NO_SAMPLES):
    start_time = time.time()
    image_to_explain = all_images[index_to_explain]

    original_output = model(image_to_explain.unsqueeze(0).to(device))
    original_prob = nn.functional.softmax(original_output, dim = 1)
    first_prediction, second_prediction = torch.topk(original_output, 2)[1][0]

    original_logodds = np.log(original_prob[0][first_prediction].cpu().detach().numpy()/original_prob[0][second_prediction].cpu().detach().numpy())
    
#     SHAP
    start_time = time.time()
    shap = shap_ex.attribute(
            image_to_explain.unsqueeze(0).to(device),
            target=first_prediction
        )
    duration_shap.append(time.time() - start_time)

    shap_2nd = shap_ex.attribute(
            image_to_explain.unsqueeze(0).to(device),
            target=second_prediction
        )
    
    modified_image = image_to_explain.clone().numpy()
    modified_mask = gen_mask(shap.squeeze(0).cpu().detach().numpy()[0] - shap_2nd.squeeze(0).cpu().detach().numpy()[0])
    modified_image[0][modified_mask==1] = 1 - modified_image[0][modified_mask==1]
    modified_image = torch.tensor(modified_image)
    modified_output = model(modified_image.unsqueeze(0).to(device))
    modified_prob = nn.functional.softmax(modified_output, dim = 1)
    modified_logodds = np.log(modified_prob[0][first_prediction].cpu().detach().numpy()/modified_prob[0][second_prediction].cpu().detach().numpy())
    logodds_score = original_logodds - modified_logodds
    
    logodds_shap.append(logodds_score)

#     DEEP
    start_time = time.time()
    deep = deep_ex.attribute(
            image_to_explain.unsqueeze(0).to(device),
            target=first_prediction
        )
    duration_deep.append(time.time() - start_time)

    deep_2nd = deep_ex.attribute(
            image_to_explain.unsqueeze(0).to(device),
            target=second_prediction
        )
    
    modified_image = image_to_explain.clone().numpy()
    modified_mask = gen_mask(deep.squeeze(0).cpu().detach().numpy()[0] - deep_2nd.squeeze(0).cpu().detach().numpy()[0])
    modified_image[0][modified_mask==1] = 1 - modified_image[0][modified_mask==1]
    modified_image = torch.tensor(modified_image)
    modified_output = model(modified_image.unsqueeze(0).to(device))
    modified_prob = nn.functional.softmax(modified_output, dim = 1)
    modified_logodds = np.log(modified_prob[0][first_prediction].cpu().detach().numpy()/modified_prob[0][second_prediction].cpu().detach().numpy())
    logodds_score = original_logodds - modified_logodds
    
    logodds_deep.append(logodds_score)
    
#     Gradient-SHAP
    start_time = time.time()
    gras = gras_ex.attribute(
            image_to_explain.unsqueeze(0).to(device),
            target=first_prediction,
            baselines=torch.zeros_like(image_to_explain.unsqueeze(0)).to(device)
        )
    duration_gras.append(time.time() - start_time)

    gras_2nd = gras_ex.attribute(
            image_to_explain.unsqueeze(0).to(device),
            target=second_prediction,
            baselines=torch.zeros_like(image_to_explain.unsqueeze(0)).to(device)
        )
    
    modified_image = image_to_explain.clone().numpy()
    modified_mask = gen_mask(gras.squeeze(0).cpu().detach().numpy()[0] - gras_2nd.squeeze(0).cpu().detach().numpy()[0])
    modified_image[0][modified_mask==1] = 1 - modified_image[0][modified_mask==1]
    modified_image = torch.tensor(modified_image)
    modified_output = model(modified_image.unsqueeze(0).to(device))
    modified_prob = nn.functional.softmax(modified_output, dim = 1)
    modified_logodds = np.log(modified_prob[0][first_prediction].cpu().detach().numpy()/modified_prob[0][second_prediction].cpu().detach().numpy())
    logodds_score = original_logodds - modified_logodds
    
    logodds_gras.append(logodds_score)
        

    infid = infidelity(model.to(device), perturb_infidelity, image_to_explain.unsqueeze(0).to(device), shap.to(device), target=first_prediction)
    infidelity_shap.append(infid.cpu().detach().numpy()[0])
    
    infid = infidelity(model.to(device), perturb_infidelity, image_to_explain.unsqueeze(0).to(device), deep.to(device), target=first_prediction)
    infidelity_deep.append(infid.cpu().detach().numpy()[0])
    
    infid = infidelity(model.to(device), perturb_infidelity, image_to_explain.unsqueeze(0).to(device), gras.to(device), target=first_prediction)
    infidelity_gras.append(infid.cpu().detach().numpy()[0])
    
    for num_one in range(10,100):
        num_ones.append(num_one)
        fid = rdt_fidelity(model.to(device), image_to_explain.unsqueeze(0).to(device), shap.to(device),  target=first_prediction, samples = 100, num_ones = num_one)
        fidelity_shap.append(fid)
        fid = rdt_fidelity(model.to(device), image_to_explain.unsqueeze(0).to(device), deep.to(device),  target=first_prediction, samples = 100, num_ones = num_one)
        fidelity_deep.append(fid)
        fid = rdt_fidelity(model.to(device), image_to_explain.unsqueeze(0).to(device), gras.to(device),  target=first_prediction, samples = 100, num_ones = num_one)
        fidelity_gras.append(fid)

    
df_infid = pd.DataFrame({'Kernel-SHAP infidelity': infidelity_shap,
                         'Kernel-SHAP logodds': logodds_shap,
                         'DEEPLIFT infidelity': infidelity_deep,
                         'DEEPLIFTn logodds': logodds_deep,
                         'Gradient-SHAP infidelity': infidelity_gras,
                         'Gradient-SHAP logodds': logodds_gras,
                         'Time Kernel-SHAP': duration_shap,
                         'Time DEEPLIFT': duration_deep,
                         'Time Gradient-SHAP': duration_gras
                        })

df_fid = pd.DataFrame({'Kernel-SHAP fidelity': fidelity_shap,
                       'DEEPLIFT fidelity': fidelity_deep,
                       'Gradient-SHAP fidelity': fidelity_gras,
                       'Num features': num_ones})
    
save_file_infid = 'result/explainer_evaluation/others_infidelity_and_logodds' + '_' + str(EXPERIMENT) + '_update2023.pickle'
save_file_fid = 'result/explainer_evaluation/others_fidelity' + '_' + str(EXPERIMENT) + '_update2023.pickle'


print("Save file to explainer_evaluation")
with open(save_file_infid, 'wb') as output:
    pickle.dump(df_infid, output)
with open(save_file_fid, 'wb') as output:
    pickle.dump(df_fid, output)