import numpy as np
import cv2
import pickle
import time
from sklearn import linear_model

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms


from fashion_model import FashionCNN 
from umap_lime import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size=100)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=100)

def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]


all_loader = torch.utils.data.DataLoader(train_set, batch_size=train_set.__len__())
all_images, all_labels = next(iter(all_loader))

start_time = time.time()
umap_sampling = UMAP_sampling(all_images)
umap_duration = time.time() - start_time
print("UMAP duration: ", umap_duration)

checkpt_file = 'pretrained/fashionCNN.pt'
print(checkpt_file)
model = FashionCNN()
model.to(device)
model.load_state_dict(torch.load(checkpt_file))

no_explanations = 1000
start_index = 0
sample_range = range(start_index, start_index + no_explanations)

sample_idx = 0
n_umap_samples = 100
n_nolime_samples = 25*n_umap_samples
n_lime_samples = 25*n_umap_samples

for images, labels in train_loader:
    for i in range(images.shape[0]):
        if sample_idx not in sample_range:
            continue
            
        print("Explaining instance: ", sample_idx)
        
        ex_img = images[i]
        
#         UMAP LIME
        start_time = time.time()
        ulime_seeds, ulime_perturbations, ulime_similarities, ulime_diff = umap_sampling.perturbations_generator(ex_img, 
                                                                                                                 no_samples = n_umap_samples, 
                                                                                                                 perturb_sigma = 10,
                                                                                                                 perturb_prob = 0.5,
                                                                                                                 constant_noise_mag = False,
                                                                                                                 mode = 'discrete',
                                                                                                                 kernel_width = 4,
                                                                                                                 low_manifold= True)
        
        ulime_perturb_targets = get_preds(model, ex_img, ulime_perturbations.float(), device, kernel_width = 4)
        clf_seed = np.abs(ulime_diff)
        ulime_explanation = fit_interpretable_model(clf_seed.reshape(n_umap_samples, -1), ulime_perturb_targets, ulime_similarities)
        ulime_explanation = ulime_explanation.reshape(28, 28)
        duration_ulime = time.time() - start_time
        print("ULIME time: ", duration_ulime)
        
#         UMAP non-embedded distance LIME
        high_similarities = [similarity_kernel(ulime_perturbations[i], ex_img, kernel_width = 4) for i in range(n_umap_samples)]
        ulime_nonembedded_explanation = fit_interpretable_model(clf_seed.reshape(n_umap_samples, -1), ulime_perturb_targets, high_similarities)
        ulime_nonembedded_explanation = ulime_nonembedded_explanation.reshape(28, 28)
        
#         LIME w/o segmentation
        start_time = time.time()
        no_u_lime_seeds, no_u_lime_perturbations, no_u_lime_similarities, no_u_lime_diff = umap_sampling.perturbations_generator(ex_img, 
                                                                                                                                 no_samples = n_nolime_samples, 
                                                                                                                                 perturb_sigma = 10,
                                                                                                                                 perturb_prob = 0.5,
                                                                                                                                 constant_noise_mag = False,
                                                                                                                                 mode = 'discrete',
                                                                                                                                 kernel_width = 4,
                                                                                                                                 low_manifold= False)
        
        no_u_lime_perturb_targets = get_preds(model, ex_img, no_u_lime_perturbations.float(), device, kernel_width = 4)
        clf_seed = np.abs(no_u_lime_diff)
        no_u_lime_explanation = fit_interpretable_model(clf_seed.reshape(n_nolime_samples, -1), no_u_lime_perturb_targets, no_u_lime_similarities)
        no_u_lime_explanation = no_u_lime_explanation.reshape(28, 28)   
        duration_no_u_lime = time.time() - start_time
        print("NLIME time: ", duration_no_u_lime)
        
#         LIME with segmentation
        start_time = time.time()
        lime_seeds, lime_perturbations, lime_similarities, segments_slic = lime_perturb_gen(ex_img, 
                                                                                             no_samples = n_lime_samples,
                                                                                             n_segments = 16, slic_compactness = 100, 
                                                                                             slic_sigma = 1, start_label = 0,
                                                                                             perturb_prob = 0.1, perturb_sigma = 1,
                                                                                             kernel_width = 4)
        
        lime_perturb_targets = get_preds(model, ex_img, lime_perturbations, device, kernel_width = 4)
        lime_explanation = fit_interpretable_model(lime_seeds, lime_perturb_targets, lime_similarities)
        lime_explanation_mask = np.zeros_like(segments_slic).astype('float64')
        for k, v in zip(np.unique(segments_slic).tolist(), lime_explanation):
            lime_explanation_mask[segments_slic == k] = v
            
        duration_lime = time.time() - start_time
        print("LIME time: ", duration_lime)
            
       
        result = [sample_idx, ex_img[0].cpu().detach().numpy(), ulime_explanation, ulime_nonembedded_explanation, no_u_lime_explanation, lime_explanation_mask]
        explain_file = 'results/explanation/ulime_explain_' + str(sample_idx) + '_.pickle'
        with open(explain_file, 'wb') as output:
            pickle.dump(result, output)
            
        time_result = [sample_idx, duration_ulime, duration_no_u_lime, duration_lime]
        time_file = 'results/time/ulime_time_' + str(sample_idx) + '_.pickle'
        with open(time_file, 'wb') as output:
            pickle.dump(time_result, output)
            
        sample_idx = sample_idx + 1