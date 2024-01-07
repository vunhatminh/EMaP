import numpy as np
import torch
import random
from sklearn import linear_model
import numpy as np
import random
import umap

class Manifold_Synthetic_Sampler(object):
    def __init__(self, 
                 data, 
                 dim = 2, 
                 std_train = 0.01,
                 random_state = 1):
        """Init function.

        Args:
            data: traning data
            dim: lower dimension
            std_train: noise to learn tangent hyperplanes
            random_state: random state for UMAP's reproducibility 
        """
        self.data = data
        self.dim = dim
        self.std_train = std_train
        self.no_training, self.num_features = self.data.shape
        self.data_min = np.min(self.data)
        self.data_max = np.max(self.data)
        self.mapper = umap.UMAP(n_components = self.dim, random_state = random_state)
        self.mapper.fit(data)
        self.pivots = None
        self.planes = None
        
        print("Finish initialization.")
        
    def transform(self, x_data):
        return self.mapper.transform(x_data)

    def inv_transform(self, low_data):
        return self.mapper.inverse_transform(low_data)

    def get_G_from_data(self):
        matA = self.mapper.transform(self.data)
        matB = self.data
        Xt = np.transpose(matA)
        XtX = np.dot(Xt,matA)
        Xty = np.dot(Xt,matB)
        matG = np.linalg.solve(XtX,Xty)
        return matG