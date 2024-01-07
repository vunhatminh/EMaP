import numpy as np
import random
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

class Manifold_Image_Sampler(object):
    def __init__(self, data, dim = 2, random_state = 1,
                labels = None,
                train_multiplier = 100,
                std_train = 0.2):
        """Init function.

        Args:
            data: traning data
        """
        self.data = data
        self.dim = dim
        self.std_train = std_train
        self.train_multiplier = train_multiplier
        self.no_training, self.channels, self.rows, self.cols = data.shape
        self.data_min = torch.min(data)
        self.data_max = torch.max(data)
        data_1d = data.reshape((self.no_training, self.channels*self.rows*self.cols))
        self.mapper = umap.UMAP(n_components = self.dim, random_state = random_state).fit(data_1d)
        self.labels = labels
        self.pivots = None
        self.planes = None
        
    def get_pivot(self, labels, no_pivots_per_label = 1, shuffle = False):
        buff = ()
        for l in torch.unique(labels):
            all_idx = (labels == l).nonzero(as_tuple=False)

            if shuffle == False:
                idx = all_idx[range(no_pivots_per_label)]
            else:
                idx = all_idx[random.sample(range(len(all_idx)),no_pivots_per_label )]

            buff = buff + (idx,)

        return torch.cat(buff, dim = 0)

    def train_pivot(self, no_pivots_per_label = 1, shuffle = False, train_labels = None,
                    perturbation_multiplier = None,
                    perturbation_std = None):
        
        if self.labels == None:
            labels = train_labels
        else:
            labels = self.labels
        
        pivots_index = self.get_pivot(labels, no_pivots_per_label, shuffle)[:,0]
        self.pivots = self.data[pivots_index]
        self.pivots_low = self.transform(self.pivots)
        hyper_planes = []
        pivots_perturbs = []
        orthor_noises = []
        plane_noises = []
        for i in pivots_index:
            mat_plane = self.get_G(self.data[i])
            Gu, Gd, Gv = np.linalg.svd(mat_plane, full_matrices=False)
            hyper_planes.append(mat_plane)
            
            if (perturbation_multiplier == None) or (perturbation_std == None):
                perturb_multiplier = self.train_multiplier
                perturb_std = self.std_train
            else:
                perturb_multiplier = perturbation_multiplier
                perturb_std = perturbation_std
                
            x_sample, y_sample = self.gen_perturbation_base(self.data[i].unsqueeze(0),
                                                   perturbation_multiplier= perturb_multiplier,
                                                   perturbation_std = perturb_std,
                                                   train = True)
            
            x_sample = np.clip(x_sample, self.data_min, self.data_max)
            
            raw_noise_1d = self.to_1d(x_sample[y_sample == 1] - x_sample[y_sample == 0])
            plane_noise = np.zeros_like(raw_noise_1d)
            for d in range(Gv.shape[0]):
                proj = np.dot(raw_noise_1d, Gv[d])
                for s in range(plane_noise.shape[0]):
                    plane_noise[s] = plane_noise[s] + proj[s]*Gv[d]        
            ortho_noise = raw_noise_1d - plane_noise
            orthor_noises.append(self.to_3d(ortho_noise))
            plane_noises.append(self.to_3d(plane_noise))
            
            x_sample_low = self.transform(x_sample[y_sample == 1].clone().detach())
            x_reverse = np.clip(np.dot(x_sample_low, mat_plane), self.data_min, self.data_max)
            x_reverse_3d = self.to_3d(x_reverse) 
            pivots_perturbs.append(x_reverse_3d)
        self.planes = hyper_planes
        self.perturbs = pivots_perturbs
        self.ortho_noise = orthor_noises
        self.plane_noise = plane_noises
        
        

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

    def gen_perturbation_base(self, X, perturbation_multiplier=10, perturbation_std = 0.3, train = True):
        all_x, all_y = [], []
        var = 0
        if train == True:
            for _ in range(perturbation_multiplier):
                perturbed_xtrain = np.random.normal(0, perturbation_std, size=X.shape)
                p_train_x = np.vstack((X, np.clip(X + perturbed_xtrain, self.data_min, self.data_max)))
                p_train_y = np.concatenate((np.zeros(X.shape[0]), np.ones(X.shape[0])))
                all_x.append(p_train_x)
                all_y.append(p_train_y)
            all_x = np.vstack(all_x)
            all_y = np.concatenate(all_y)
            return all_x, all_y
        else:
            for _ in range(perturbation_multiplier):
                perturbed_xtrain = np.random.normal(0, perturbation_std, size=X.shape)
                p_train_x = np.clip(X + perturbed_xtrain, self.data_min, self.data_max)
                all_x.append(p_train_x)
                var = var + np.var(p_train_x.numpy() - X.numpy())
            all_x = np.vstack(all_x)
            return all_x

    def to_1d(self, data):
        return data.reshape((data.shape[0], self.channels*self.rows*self.cols))

    def to_3d(self, data):
        return data.reshape((data.shape[0], self.channels, self.rows, self.cols))

    def get_G(self, x):
        x_sample = self.gen_perturbation_base(x.unsqueeze(0),
                                            perturbation_multiplier=self.train_multiplier,
                                            perturbation_std = self.std_train,
                                            train = False)

        matA = self.transform(x_sample)
        matB = self.to_1d(x_sample)
        Xt = np.transpose(matA)
        XtX = np.dot(Xt,matA)
        Xty = np.dot(Xt,matB)
        matG = np.linalg.solve(XtX,Xty)

        return matG
    
    def get_x_noise_low_plane(self, x, mat_plane,
                     perturbation_multiplier = 1, std_value = 0.1):
        
        x_sample, y_sample = self.gen_perturbation_base(x.unsqueeze(0),
                                               perturbation_multiplier=perturbation_multiplier,
                                               perturbation_std = std_value,
                                               train = True)
        x_sample = np.clip(x_sample, self.data_min, self.data_max)
        x_sample_low = self.transform(x_sample[y_sample == 1].clone().detach())
        x_reverse = np.clip(np.dot(x_sample_low, mat_plane), self.data_min, self.data_max)
        x_reverse_3d = self.to_3d(x_reverse) 
        x_sample[y_sample == 1] = x_reverse_3d
        x_ori_low = self.transform(x_sample[y_sample != 1].clone().detach())
        geo_distance = np.linalg.norm(x_ori_low - x_sample_low, axis=1)
        return x_reverse_3d, geo_distance
    
    def get_hyperplane_perturbations(self, x, no_samples = 10, std_x = 0.1):
        x_G = self.get_G(x)
        x_per, x_d = self.get_x_noise_low_plane(x, x_G, no_samples, std_x)
        x_low = np.repeat(self.transform(x.unsqueeze(0)), self.perturbs[0].shape[0], axis = 0)
        perturb_d = []
        for perturb in self.perturbs:
            perturb_low = self.transform(perturb)
            perturb_d.append(np.linalg.norm(perturb_low - x_low, axis=1))
        perturb_d.append(x_d)
        perturbations = self.perturbs.copy()
        perturbations.append(x_per)
        perturbations_np = np.vstack(perturbations)
        perturb_d_np = np.concatenate(perturb_d)
        x_dup = torch.cat(perturbations_np.shape[0]*[x.unsqueeze(0)])
        return x_dup, perturbations_np, perturb_d_np
