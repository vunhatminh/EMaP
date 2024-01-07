import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from skimage.segmentation import slic, quickshift
from skimage.segmentation import mark_boundaries

from sklearn import linear_model

import umap

def similarity_kernel(v1,v2,kernel_width = 1):
    l2_dist = np.linalg.norm(v1 - v2)
    return np.exp(- (l2_dist**2) / (kernel_width**2))

def torch_img_to_np(torch_img):
    np_img = torch_img.numpy()
    np_img = np.swapaxes(np_img,0,1)
    np_img = np.swapaxes(np_img,1,2)
    return np_img

def np_img_to_torch(np_img):
    torch_img = np.swapaxes(np_img,2,1)
    torch_img = np.swapaxes(torch_img,1,0)
    return torch.from_numpy(torch_img)

def segment_img(torch_img, n_segments=10, compactness=2, sigma=1, start_label=0):
    np_img = torch_img_to_np(torch_img)
    segments_slic = slic(np_img, n_segments=n_segments, compactness=compactness,
                         sigma=sigma, start_label=start_label)
    return np_img, np.repeat(np_img, 3, axis=2), segments_slic

def lime_seed_generator(xai_dim, p = 0.9):
    return list(np.random.choice(2, xai_dim, p=[1-p, p]))

def get_perturb_mask(segments_slic ,p = 0.9):
    xai_dim = len(np.unique(segments_slic).tolist())
    seed = lime_seed_generator(xai_dim, p = p)
    perturb_mask = np.zeros_like(segments_slic)
    for key, val in zip(np.unique(segments_slic).tolist(), seed):
        perturb_mask[segments_slic == key] = val
    return seed, 1 - perturb_mask

def lime_noise_generator(x, segments_slic, p = 0.9, sigma = 1):
    lime_seed, perturb_mask = get_perturb_mask(segments_slic ,p = p) 
    noise = np.random.normal(0, sigma, x.shape)
    return lime_seed, noise*perturb_mask[:,:,None]

def lime_perturb_gen(torch_img, 
                     no_samples = 1,
                     n_segments=10, slic_compactness=2, slic_sigma=1, start_label=0,
                     perturb_prob = 0.1, perturb_sigma = 1,
                     data_min = 0.0, data_max = 1.0,
                     kernel_width = 1):
    
    np_img, np_3D_img, segments_slic = segment_img(torch_img, 
                                                   n_segments=n_segments, 
                                                   compactness=slic_compactness, 
                                                   sigma=slic_sigma, 
                                                   start_label=0)
    
    xai_dim = len(np.unique(segments_slic).tolist())
    seeds = np.zeros((no_samples, xai_dim))
    perturbations = torch.cat(no_samples*[torch_img.unsqueeze(0)])
    for s in range(no_samples):
        seeds[s], noise = lime_noise_generator(np_img, segments_slic ,
                                               p = 1 - perturb_prob, 
                                               sigma = perturb_sigma)

        perturbations[s] =  torch.clip(perturbations[s] + np_img_to_torch(noise),
                                       min = data_min, max = data_max)
        
        similarities = [similarity_kernel(perturbations[i], torch_img, kernel_width = kernel_width) for i in range(no_samples)]
    
    return seeds, perturbations, similarities, segments_slic


def get_preds(f, x, perturbations, device, kernel_width = 1):
    
    original_output = f(x.unsqueeze(0).to(device))
    original_prediction = torch.max(original_output, 1)[1]
    perturb_outputs = f(perturbations.to(device))
    perturb_prediction = torch.max(perturb_outputs, 1)[1]
    probs = nn.functional.softmax(perturb_outputs, dim = 1)
    perturb_targets = probs[:,original_prediction.item()].cpu().detach().numpy()
    
    return perturb_targets

def fit_interpretable_model(seeds, perturb_targets, similarities, interpretable_model = "Ridge"):
    if interpretable_model == "Ridge":
        clf = linear_model.Ridge(alpha = 100)
        clf.fit(seeds, perturb_targets, sample_weight=similarities)
        explanation = clf.coef_
        explanation = (explanation - explanation.min())/(explanation.max() - explanation.min())
    else:
        explanation = None
    return explanation

class UMAP_sampling(object):
    def __init__(self, data, dim = 2, random_state = 1):
        """Init function.

        Args:
            data: traning data
        """
        self.dim = dim
        self.no_training, self.channels, self.rows, self.cols = data.shape
        data_1d = data.reshape((self.no_training, self.channels*self.rows*self.cols))
        self.mapper = umap.UMAP(n_components = self.dim, random_state = random_state).fit(data_1d)
    
    def transform(self, data):
        no_data, channels, rows, cols = data.shape
        return self.mapper.transform(data.reshape(no_data, channels*rows*cols))
    
    def inv_transform(self, low_data):
        no_data, low_dim = low_data.shape
        
        assert low_dim == self.dim, "Mismatched dimension"
        
        inv_1d_imgs = self.mapper.inverse_transform(low_data)
        inv_imgs_np = inv_1d_imgs.reshape((no_data, self.channels, self.rows, self.cols))
        inv_imgs = torch.from_numpy(inv_imgs_np)
        
        return inv_imgs
    
    def seed_generator(self, dim):
        assert isinstance(dim,int), "dim need to be integer"
        return list(np.random.rand(dim))

    def noise_generator(self, seed, sigma = 1):
        noise = []
        for s in seed:
            std = np.abs(1-s)
            noise.append(np.random.normal(0, sigma*std))
        return np.asarray(noise)
    
    def seed_to_int(self, seeds, delta = 0.5):
        seeds_int = seeds.copy()
        seeds_int[seeds > delta]  = 1
        seeds_int[seeds <= delta]  = 0
        return seeds_int
    
    def perturbation_generator(self, x, no_samples, sigma = 1, mode = 'continuous', 
                               perturb_prob = 0.1,
                               constant_noise_mag = False):
        rand_seed = self.seed_generator(x.nelement()*no_samples)
        if mode == 'discrete':
            seed = self.seed_to_int(np.asarray(rand_seed), delta = perturb_prob)
            seed = list(seed)
        else:
            seed = rand_seed
        
        if constant_noise_mag == True:
            seed = list(np.zeros_like(np.asarray(rand_seed)))
    
        perturbation = self.noise_generator(seed, sigma)
        output_shape = (no_samples,) + tuple(x.shape)
        return np.asarray(seed).reshape(output_shape), perturbation.reshape(output_shape)

    def perturbations_generator(self, x, no_samples, data_min = None, data_max = None, 
                                perturb_sigma = 1,
                                mode = 'continuous', perturb_prob = 0.1,
                                constant_noise_mag = False,
                                kernel_width = 4,
                                embedded_distance = True,
                                low_manifold = True):

        if data_min == None:
            data_min = torch.min(x)
        if data_max == None:
            data_max = torch.max(x)

        seeds, noises = self.perturbation_generator(x, no_samples, perturb_sigma, mode, perturb_prob, constant_noise_mag)
        repeat_shape = (no_samples,) + tuple(np.ones(x.ndim). astype(int))
        
        perturbations = torch.clip(x.repeat(repeat_shape) + noises, min = data_min, max = data_max)
        
        if low_manifold == True:
            
            embeded_perturbations = self.transform(perturbations)
            recv_perturbations = self.inv_transform(embeded_perturbations)
            recv_perturbations = torch.clip(recv_perturbations, min = data_min, max = data_max)

            if embedded_distance == True:
                base_embeded = self.transform(x.unsqueeze(0))
                similarities = [similarity_kernel(embeded_perturbations[i], base_embeded, kernel_width = kernel_width) for i in range(no_samples)]
            else:
                similarities = [similarity_kernel(recv_perturbations[i], x, kernel_width = kernel_width) for i in range(no_samples)]

            _, channels, rows, cols = recv_perturbations.shape
            repeat_shape = (no_samples,) + tuple(np.ones(x.ndim). astype(int))
            repeat_x = x.repeat(repeat_shape)
            diff = recv_perturbations - repeat_x
            
            return seeds, recv_perturbations.float(), similarities, diff
        
        else:
            
            similarities = [similarity_kernel(perturbations[i], x, kernel_width = kernel_width) for i in range(no_samples)]
            _, channels, rows, cols = perturbations.shape
            repeat_shape = (no_samples,) + tuple(np.ones(x.ndim). astype(int))
            repeat_x = x.repeat(repeat_shape)
            diff = perturbations - repeat_x
            return seeds, perturbations.float(), similarities, diff
