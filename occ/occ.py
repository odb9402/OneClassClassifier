import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
import sklearn
import numpy as np
import pydot
import graphviz
import matplotlib.pyplot as plt
import matplotlib
import scipy
import pandas as pd
import random
import sys

from scipy import io

from sklearn.model_selection import train_test_split as train_test_split
from sklearn.decomposition import PCA as PCA
from sklearn.decomposition import KernelPCA as kPCA
from sklearn.preprocessing import Normalizer
from sklearn.base import BaseEstimator, ClassifierMixin

from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from numpy import ndarray
from typing import Any
from typing import Optional

from occ.models.twolineAE import twolineAE
from occ.models.ensemble import ensemble
from occ.models.ocnn import ocnn
from occ.models.AutoEncoderODD import AutoEncoderODD
from occ.models.VAE_ODD import VAE_ODD 
from occ.models.deep_SVDD import deep_SVDD 

matplotlib.style.use("ggplot")
tf.test.is_gpu_available()
#gpus = tf.config.experimental.list_physical_devices('GPU')

class occ():
    """
    One-class classifier for outlier detection.
    
    Attributes:
        data
        model
        X
        Y
        score
        X_proj
    
    Methods:
    
    """
    def __init__(self):
        self.data = None
        self.model = None
        self.X = None
        self.Y = None
        self.score = None
        #self.X_train = None
        #self.Y_train = None
        #self.X_test = None
        #self.Y_test = None
        
    def load_data_mat(self, file_name):
        """
        Load data from .mat file.
        Note that data from ODDS contains a lot of .mat data including known anomalies(Y)
        """
        # type: (str) -> None
        self.data = scipy.io.loadmat(file_name)
        self.X = self.data['X']
        self.Y = self.data['y']
        self.X_proj = occ.manipulation(self.X)
        #self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.15)
    
    def load_data_csv(self, file_name, Y=False, **kwargs):
        """
        The CSV file should be formatted as:
         X0, X1, ..., Xn, Y
        
        """
        data = pd.read_csv(file_name, header=None, **kwargs)
        self.X = data[range(data.shape[1]-1)].values
        if Y:
            self.Y = data[data.shape[1]-1].to_numpy().reshape([data.shape[0],1])
        else:
            pass

    def load_data_npz(self, file_name, Y=False):
        self.X = np.load(file_name)
        if not Y:
            self.Y = np.load(Y)
        else:
            pass
        
    def train(self, model='ocsvm', data=None, sampling=False, **kwargs):
        # type: (str, Optional[Any], bool, **Any) -> None
        """
        
        :param sampling: (float) Proportion of sampling. If the raw data size is too large,
            we can use the sampled datasets.
        
        """
        if type(data) != np.ndarray:
            if data == None:
                data = self.X
        if sampling != False:
            data = occ.sampling_X(self.X, rate=sampling)
        kernel_set = 'poly'
        gamma_set = 'scale'
        epochs = 10
        batch_size = 50
        nu = 0.1
        hidden_neurons = None 
        known_normal = False
        kernel_epochs = 50
        radius_epochs = 100
        neighbors = 10

        for k, v in kwargs.items():
            if 'kernel' == k:
                kernel_set = v
            elif 'gamma' == k:
                gamma_set = v
            elif 'epochs' == k:
                epochs = v
            elif 'nu' == k:
                nu = v
            elif 'batch_size' == k:
                batch_size = v
            elif 'hidden_neurons' == k:
                hidden_neurons = v
            elif 'known_normal' == k:
                known_normal = v
            elif 'kernel_epochs' == k:
                kernel_epochs = v
            elif 'radius_epochs' == k:
                radius_epochs = v
            elif 'neighbors' == k:
                neighbors = v
                
        if model=='ocsvm':
            self.model = sklearn.svm.OneClassSVM(gamma=gamma_set, kernel=kernel_set, nu=nu)
        elif model=='ocnn':
            self.model = ocnn(len(data[0]), epochs=epochs, nu=nu, batch_size=batch_size)
        elif model == 'ensemble':
            self.model = ensemble(nu=nu)
        elif model == 'isoForest':
            self.model = sklearn.ensemble.IsolationForest(contamination=nu)
        elif model == 'autoEncoder':
            self.model = AutoEncoderODD(nu=nu, hidden_neurons=hidden_neurons, epochs=epochs, batch_size=batch_size)
        elif model == 'vae':
            self.model = VAE_ODD(nu=nu, hidden_neurons=hidden_neurons, epochs=epochs, batch_size=batch_size)
        elif model == 'deepsvdd':
            self.model = deep_SVDD(nu=nu, known_normal=known_normal, hidden_neurons=hidden_neurons
                                ,kernel_epochs=kernel_epochs, radius_epochs=radius_epochs, batch_size=batch_size)
        elif model == 'knn':
            self.model = KNN(contamination=nu, n_neighbors=neighbors)
        elif model == 'twolineAE':
            self.model = twolineAE(nu=nu, hidden_neurons=hidden_neurons, epochs=epochs, batch_size=batch_size)
        else:
            print("There is no such model type {}".format(model))
        
        data = self.select_data(data, **kwargs)

        self.model.fit(data)
    
    def predict(self, data=None, **kwargs):
        # type: (Optional[Any], **Any) -> ndarray
        if type(data) != np.ndarray:
            if data == None:
                data = self.X
        data = self.select_data(data, **kwargs)
        return self.model.predict(data).reshape(len(data),1)
    
    def get_score(self, data=None, **kwargs):
        if type(data) != np.ndarray:
            if data == None:
                data = self.X
        data = self.select_data(data, **kwargs)
        return self.model.score_samples(data).reshape(len(data),1)
    
    def export_csv(self, file_name, score):
        if type(self.Y) != np.ndarray:
            if self.Y == None:
                array = np.concatenate((self.X, score), axis=1)
        else:
            array = np.concatenate((self.X, self.Y, score), axis=1)
        pd.DataFrame(array).to_csv(file_name, header=None, index=False)
    
    def export_outliers(self, file_name, predictions):
        def export(array):
            pd.DataFrame(array[np.where(predictions == -1)[0]]).to_csv(file_name, header=None, index=False)
            
        if type(self.Y) != np.ndarray:
            if self.Y == None:
                export(self.X)
        else:
            X = self.X
            Y = self.Y
            export(np.concatenate((X,Y), axis=1))
    
    @staticmethod
    def select_data(data, **kwargs):
        # type: (ndarray, **Any) -> ndarray
        norm = False
        manipulate = False
        for k, v in kwargs.items():
            if 'norm' == k:
                norm = v
            if 'manipulate' == k:
                manipulate = v
        if norm:
            data = occ.norm(data)
        if manipulate:
            data = occ.manipulation(data)
        return data
    
    @staticmethod
    def manipulation(data, **kwargs):
        # type: (ndarray, **Any) -> ndarray
        method = 'pca'
        dim = 3
        for k, v in kwargs.items():
            if 'method' == k:
                method = v
            if 'dim' == k:
                dim = v
        if method == 'pca':
            projection = PCA(n_components=dim)
        projection.fit(data)
        return projection.transform(data)
        
    @staticmethod
    def show_projection(data, label=None, **kwargs):
        # type: (ndarray, ndarray, **Any) -> None
        size = 25
        cmap = 'viridis'
        norm = False
        title = None
        save_file = None
        
        for k, val in kwargs.items():
            if 'title' == k:
                title = val
            elif 'markersize' == k:
                size = val
            elif 'cmap' == k:
                cmap = val
            elif 'norm' == k:
                norm = val
            elif 'save_file' == k:
                save_file = val
            
        data = occ.select_data(data, **kwargs)
        data_proj = occ.manipulation(data, method='pca', dim=2)
        data_proj_t = data_proj.transpose()
        ax, fig = plt.subplots(figsize=(10,10))
        ax = plt.scatter(data_proj_t[0], data_proj_t[1], c=label, s=size, marker='.')
        ax = plt.colorbar()
        ax = plt.set_cmap(cmap)
        if title != None:
            plt.title(title)
        if save_file != None:
            plt.savefig(save_file, dpi=300)
        plt.show()
    
    @staticmethod
    def norm(data):
        # type: (ndarray) -> ndarray
        norm = sklearn.preprocessing.Normalizer(norm='l2', copy=True).fit(data)
        return norm.transform(data)
    
    @staticmethod
    def proportion(data):
        return np.where(data < 0)

    @staticmethod 
    def sampling_X(X, rate=0.1):
        idx = list(range(len(X)))
        random.shuffle(idx)
        return X[idx[:int(len(X)*rate)]] 