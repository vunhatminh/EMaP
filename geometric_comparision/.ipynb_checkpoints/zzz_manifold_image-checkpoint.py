import numpy as np
import random
from sklearn import linear_model
import umap


class Manifold_Image_Sampler(object):
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
        self.data_min = np.min(self.data)
        self.data_max = np.max(self.data)
        data_1d = data.reshape((self.no_training, self.channels*self.rows*self.cols))
        self.mapper = umap.UMAP(n_components = self.dim, random_state = random_state)
        self.mapper.fit(data_1d)
        self.labels = labels
        self.pivots = None
        self.planes = None
        
    def get_pivots(self, labels, no_pivots_per_label = 1, shuffle = False, target_labels = None):        
        if target_labels == None:
            target_labels = np.unique(labels)
        
        buff = []
        for l in target_labels:
            all_idx = (labels == l).nonzero()[0]

            if shuffle == False:
                idx = all_idx[range(no_pivots_per_label)]
            else:
                idx = all_idx[random.sample(range(len(all_idx)),no_pivots_per_label )]
            
            for i in idx:
                buff.append(i)
            
        self.pivots = self.data[buff].copy()
#         return buff, self.pivots
    
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
        matA = self.mapper.transform(self.to_1d(x_sample))
#         matA = self.transform(x_sample)
        matB = self.to_1d(x_sample)
        Xt = np.transpose(matA)
        XtX = np.dot(Xt,matA)
        Xty = np.dot(Xt,matB)
        matG = np.linalg.solve(XtX,Xty)
        return matG
    
    def get_G_from_pivots(self):
        matA = self.mapper.transform(self.to_1d(self.pivots))
        matB = self.to_1d(self.pivots)
        Xt = np.transpose(matA)
        XtX = np.dot(Xt,matA)
        Xty = np.dot(Xt,matB)
        matG = np.linalg.solve(XtX,Xty)
        return matG
    
    def to_1d(self, data):
        return data.reshape((data.shape[0], self.channels*self.rows*self.cols))

    def to_3d(self, data):
        return data.reshape((data.shape[0], self.channels, self.rows, self.cols))
