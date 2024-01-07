import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

# from sklearn import linear_model
import random
import umap

class Manifold_Image_Sampler(object):
    def __init__(self, data, dim = 2, random_state = 1,
                no_planes_per_class = 1,
                hyper_planes = False,
                labels = None,
                # no_pivots_per_label = 1,
                # shuffle = False,
                train_multiplier = 10,
                std_train = 0.1):
        """Init function.

        Args:
            data: traning data
        """
        self.dim = dim
        self.std_train = std_train
        self.train_multiplier = train_multiplier
        self.no_training, self.channels, self.rows, self.cols = data.shape
        data_1d = data.reshape((self.no_training, self.channels*self.rows*self.cols))
        self.mapper = umap.UMAP(n_components = self.dim, random_state = random_state).fit(data_1d)
        self.labels = labels
        self.pivots = None
        self.planes = None

    def train_pivot(self, no_pivots_per_label = 1, shuffle = False,):
        pivots_index = get_pivot(labels, no_pivots_per_label, shuffle)
        self.pivots = data[pivots_index]
        hyper_planes = []
        for i in pivots_index:
            hyper_planes.append(self.get_G(data[i], perturbation_multiplier = train_multiplier, std_value = std_train))
        self.planes = hyper_planes
            
    
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

    def gen_perturbation_base(self, X, perturbation_multiplier=10, perturbation_std = 0.3, train = True, data_min = 0.0, data_max = 1.0):
        all_x, all_y = [], []
        var = 0
        if train == True:
            for _ in range(perturbation_multiplier):
                perturbed_xtrain = np.random.normal(0, perturbation_std, size=X.shape)
                p_train_x = np.vstack((X, X + perturbed_xtrain))
                p_train_y = np.concatenate((np.zeros(X.shape[0]), np.ones(X.shape[0])))

                all_x.append(p_train_x)
                all_y.append(p_train_y)
                var = var + np.var(perturbed_xtrain)
            all_x = np.vstack(all_x)
            all_y = np.concatenate(all_y)

            return all_x, np.sqrt(var/perturbation_multiplier), all_y
        else:
            for _ in range(perturbation_multiplier):
                perturbed_xtrain = np.random.normal(0, perturbation_std, size=X.shape)
                p_train_x = np.clip(X + perturbed_xtrain, data_min, data_max)
                all_x.append(p_train_x)
                var = var + np.var(p_train_x.numpy() - X.numpy())
            all_x = np.vstack(all_x)

            return all_x, np.sqrt(var/perturbation_multiplier)

    def to_1d(self, data):
        return data.reshape((data.shape[0], self.channels*self.rows*self.cols))

    def to_3d(self, data):
        return data.reshape((data.shape[0], self.channels, self.rows, self.cols))

    def get_G(self, x):
        x_sample, std_sample = self.gen_perturbation_base(x.unsqueeze(0),
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

    def get_pivot_perturbation(self, x, )

    def get_hyperplane_perturbation(self, x, hyperplane,
                                   per_type = 'low_plane',
                                   perturbation_multiplier = 1, std_value = 0.1,
                                   data_min = 0.0, data_max = 1.0,
                                   on_pivots = False):

        matG = hyperplane
        x_sample, std_sample, y_sample = self.gen_perturbation_base(x.unsqueeze(0),
                                                   perturbation_multiplier=perturbation_multiplier,
                                                   perturbation_std = std_value,
                                                   train = True)
        x_sample = np.clip(x_sample, data_min, data_max)

        if per_type == 'low_plane':
            x_sample_low = self.transform(torch.tensor(x_sample[y_sample == 1]))
            x_reverse = np.clip(np.dot(x_sample_low, matG), data_min, data_max)
            x_reverse_3d = self.to_3d(x_reverse)
            std_real = np.var(x_reverse_3d - x_sample[y_sample != 1])
            x_sample[y_sample == 1] = x_reverse_3d


            x_ori_low = self.transform(torch.tensor(x_sample[y_sample != 1]))
            d = np.linalg.norm(x_ori_low - x_sample_low, axis=1)
            return x_sample, std_real, d, y_sample

        else:
            Gu, Gd, Gv = np.linalg.svd(matG, full_matrices=False)

            raw_noise = x_sample[y_sample == 1] - x_sample[y_sample != 1]
            raw_noise_1d = umap_sampling.to_1d(raw_noise)

            plane_noise = np.zeros_like(raw_noise_1d)
            for d in range(Gv.shape[0]):
                proj = np.dot(raw_noise_1d, Gv[d])
                for s in range(plane_noise.shape[0]):
                    plane_noise[s] = plane_noise[s] + proj[s]*Gv[d]

            ortho_noise = raw_noise_1d - plane_noise
            ortho_noise_3d = umap_sampling.to_3d(ortho_noise)
            plane_noise_3d = umap_sampling.to_3d(plane_noise)
            if per_type == 'on_plane':
                x_sample[y_sample == 1] = np.clip(x_sample[y_sample == 1] - ortho_noise_3d, data_min, data_max)
                std_real = np.var(x_sample[y_sample == 1] - x_sample[y_sample != 1])
            elif per_type == 'perp_plane':
                x_sample[y_sample == 1] = np.clip(x_sample[y_sample == 1] - plane_noise_3d, data_min, data_max)
                std_real = np.var(x_sample[y_sample == 1] - x_sample[y_sample != 1])

            x_sample_low = self.transform(torch.tensor(x_sample[y_sample == 1]))
            x_ori_low = self.transform(torch.tensor(x_sample[y_sample != 1]))
            d = np.linalg.norm(x_ori_low - x_sample_low, axis=1)
            return x_sample, std_real, d, y_sample
