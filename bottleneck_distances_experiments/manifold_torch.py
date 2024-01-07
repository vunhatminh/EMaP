import numpy as np
import torch
import random
from sklearn import linear_model
import umap


class Manifold_Image_Sampler(object):
    # Expect torch data
    def __init__(self, data, dim = 2, random_state = 1,
                labels = None,
                std_train = 0.2):
        """Init function.

        Args:
            data: traning data
        """
        self.data = data
        self.dim = dim
        self.std_train = std_train
        self.no_training, self.channels, self.rows, self.cols = self.data.shape
        self.data_min = torch.min(self.data)
        self.data_max = torch.max(self.data)
        data_1d = data.reshape((self.no_training, self.channels*self.rows*self.cols))
        self.mapper = umap.UMAP(n_components = self.dim, random_state = random_state)
        self.mapper.fit(data_1d)
        self.labels = labels
        self.pivots = None
        self.planes = None
        
    def get_pivots(self, labels, no_pivots_per_label = 1, shuffle = False, target_labels = None):        
        if target_labels == None:
            target_labels = torch.unique(labels)
        
        buff = []
        for l in target_labels:
            all_idx = (labels == l).nonzero(as_tuple=False)

            if shuffle == False:
                idx = all_idx[range(no_pivots_per_label)]
            else:
                idx = all_idx[random.sample(range(len(all_idx)),no_pivots_per_label )]
            
            for i in idx:
                buff.append(i)
        buff = [buff[i].cpu().detach().numpy()[0] for i in range(len(buff))]
            
        self.pivots = self.data[buff].clone()
        return buff
    
    def transform(self, x_data):
#         Expect [N,c,l,w] data 
        no_data, channels, rows, cols = x_data.shape
        return self.mapper.transform(x_data.reshape(no_data, channels*rows*cols))

    def inv_transform(self, low_data):
#         Expect [N,d] data
        no_data, low_dim = low_data.shape

        assert low_dim == self.dim, "Mismatched dimension"

        inv_1d_imgs = self.mapper.inverse_transform(low_data)
        inv_imgs = inv_1d_imgs.reshape((no_data, self.channels, self.rows, self.cols))

        return inv_imgs
    
    def get_G_from_samples(self, x_sample):
        matA = self.mapper.transform(self.to_1d(x_sample.cpu().detach().numpy()))
        matB = self.to_1d(x_sample.cpu().detach().numpy())
        Xt = np.transpose(matA)
        XtX = np.dot(Xt,matA)
        Xty = np.dot(Xt,matB)
        matG = np.linalg.solve(XtX,Xty)
        return matG
    
    def get_G_from_pivots(self):
        matA = self.mapper.transform(self.to_1d(self.pivots.cpu().detach().numpy()))
        matB = self.to_1d(self.pivots.cpu().detach().numpy())
        Xt = np.transpose(matA)
        XtX = np.dot(Xt,matA)
        Xty = np.dot(Xt,matB)
        matG = np.linalg.solve(XtX,Xty)
        return matG
    
    def to_1d(self, data):
        return data.reshape((data.shape[0], self.channels*self.rows*self.cols))

    def to_3d(self, data):
        return data.reshape((data.shape[0], self.channels, self.rows, self.cols))

    
class Manifold_Tabular_Sampler(object):
    # Expect torch data
    def __init__(self, data, dim = 2, random_state = 1,
                labels = None,
                std_train = 0.2):
        """Init function.

        Args:
            data: traning data
        """
        self.data = data
        self.dim = dim
        self.std_train = std_train
        self.no_training, self.num_features = self.data.shape
        self.data_min = torch.min(self.data)
        self.data_max = torch.max(self.data)
        self.mapper = umap.UMAP(n_components = self.dim, random_state = random_state)
        self.mapper.fit(data)
        self.labels = labels
        self.pivots = None
        self.planes = None
        
    def get_pivots(self, labels, no_pivots_per_label = 1, shuffle = False, target_labels = None):        
        if target_labels == None:
            target_labels = torch.unique(labels)
        
        buff = []
        for l in target_labels:
            all_idx = (labels == l).nonzero(as_tuple=False)

            if shuffle == False:
                idx = all_idx[range(no_pivots_per_label)]
            else:
                idx = all_idx[random.sample(range(len(all_idx)),no_pivots_per_label )]
            
            for i in idx:
                buff.append(i)
        buff = [buff[i].cpu().detach().numpy()[0] for i in range(len(buff))]
            
        self.pivots = self.data[buff].clone()
        return buff
    
    def transform(self, x_data):
        return self.mapper.transform(x_data)

    def inv_transform(self, low_data):
#         Expect [N,d] data
        return self.mapper.inverse_transform(low_data)

    
    def get_G_from_samples(self, x_sample):
        matA = self.mapper.transform(x_sample.cpu().detach().numpy())
        matB = x_sample.cpu().detach().numpy()
        Xt = np.transpose(matA)
        XtX = np.dot(Xt,matA)
        Xty = np.dot(Xt,matB)
        matG = np.linalg.solve(XtX,Xty)
        return matG
    
    def get_G_from_pivots(self):
        matA = self.mapper.transform(self.pivots.cpu().detach().numpy())
        matB = self.pivots.cpu().detach().numpy()
        Xt = np.transpose(matA)
        XtX = np.dot(Xt,matA)
        Xty = np.dot(Xt,matB)
        matG = np.linalg.solve(XtX,Xty)
        return matG