from occ.occ import occ 
import numpy as np
import pandas as pd

class occ_flat(occ):
    """
    If we want to use a tensor as the input of occ algorithms, we can use occ_flat.
    For example, if you apply a occ model for an image dataset such as CIFAR10,
    the shape of targets is (batch_size, W, H, C). As long as the implemented algorithm
    does not use convolutional operation, the data should be transformed into
    (batch_size, features). occ_flat can apply for those problems.

    The input data of normal occ : (batch_size, features)
    The input data of occ_flat : (batch_size, ?, ..., ?) -> (batch_size, features)
    """
    
    def __init__(self, X):
        super().__init__()
        self.raw_X = X
        self.X = np.array(list(map(np.ravel, X)))

    def load_data_csv(self, file_name, Y=False, **kwargs):
        super().load_data_csv(file_name, Y=Y, **kwargs)
        self.raw_X = self.X
        self.X = np.array(list(map(np.ravel, self.X)))

    def export_score(self, file_name, **kwargs):
        if type(self.Y) != np.ndarray:
            if self.Y == None:
                array = np.concatenate((self.raw_X, self.get_score(**kwargs)), axis=1)
        else:
            array = np.concatenate((self.raw_X, self.Y, self.get_score(**kwargs)), axis=1)
        pd.DataFrame(array).to_csv(file_name, header=None, index=False)
    
    def export_csv(self, file_name, score):
        if type(self.Y) != np.ndarray:
            if self.Y == None:
                array = np.concatenate((self.raw_X, score), axis=1)
        else:
            array = np.concatenate((self.raw_X, self.Y, score), axis=1)
        pd.DataFrame(array).to_csv(file_name, header=None, index=False)
