import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
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

from abc import *

from scipy import io

from sklearn.model_selection import train_test_split as train_test_split
from sklearn.decomposition import PCA as PCA
from sklearn.decomposition import KernelPCA as kPCA
from sklearn.preprocessing import Normalizer
from sklearn.base import BaseEstimator, ClassifierMixin

from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.lscp import LSCP
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE 
from suod.models.base import SUOD
from numpy import ndarray
from typing import Any
from typing import Optional
from tensorflow.python.framework.ops import EagerTensor

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
        if Y:
            self.X = data[range(data.shape[1]-1)].values
            self.Y = data[data.shape[1]-1].to_numpy().reshape([data.shape[0],1])
        else:
            self.X = data[range(data.shape[1])].values
    
        
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
        hidden_neurons = [64,32,32,64]
        
        for k, v in kwargs.items():
            if 'kernel' == k:
                kernel_set = v
            if 'gamma' == k:
                gamma_set = v
            if 'epochs' == k:
                epochs = v
            if 'nu' == k:
                nu = v
            if 'batch_size' == k:
                batch_size = v
            if 'hidden_neurons' == k:
                hidden_neurons = v
                
        if model=='ocsvm':
            self.model = sklearn.svm.OneClassSVM(gamma=gamma_set, kernel=kernel_set, nu=nu)
        elif model=='ocnn':
            self.model = ocnn(len(data[0]), epochs=epochs, nu=nu, batch_size=batch_size)
        elif model == 'ensemble':
            self.model = ensemble(nu=nu)
        elif model == 'isolationForest':
            self.model = sklearn.ensemble.IsolationForest(contamination=nu)
        elif model == 'autoEncoder':
            self.model = AutoEncoderOOD(nu=nu, hidden_neurons=hidden_neurons, epochs=epochs)
        elif model == 'vae':
            self.model = VAE_ODD(nu=nu, hidden_neurons=hidden_neurons, epochs=epochs)
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
        for k, val in kwargs.items():
            if 'title' == k:
                title = val
            elif 'markersize' == k:
                size = val
            elif 'cmap' == k:
                cmap = val
            elif 'norm' == k:
                norm = val
            
        data = occ.select_data(data, **kwargs)
        data_proj = occ.manipulation(data, method='pca', dim=2)
        data_proj_t = data_proj.transpose()
        ax, fig = plt.subplots(figsize=(10,10))
        ax = plt.scatter(data_proj_t[0], data_proj_t[1], c=label, s=size, marker='.')
        ax = plt.colorbar()
        ax = plt.set_cmap(cmap)
        if title != None:
            plt.title(title)
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
    
    
class custom_hinge_loss(tf.keras.losses.Loss):
    def __init__(self, nu, r):
        super().__init__()
        self.nu = nu
        self.r = r
        
    @tf.function
    def call(self, y_true, y_pred):
        y_pred_prev = y_pred
        loss_val = custom_hinge_loss.quantile_loss(self.r, y_pred, self.nu)
        self.r = tfp.stats.percentile(tf.reduce_max(y_pred_prev, axis=1), 100 * self.nu)
        return loss_val
    
    @tf.function
    def quantile_loss(r, y, nu):
        """
        3rd term in Eq (4) of the original paper
        :param r: bias of hyperplane
        :param y: data / output we're operating on
        :param nu: parameter between [0, 1] controls trade off between maximizing the distance of the hyperplane from
            the origin and the number of data points permitted to cross the hyper-plane (false positives) (default 1e-2)
        :return: the loss function value
        """
        
        return (1 / nu) * tf.keras.backend.mean(tf.keras.backend.maximum(0.0, r - y), axis=-1)


class abstract_occ_model(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X):
        pass 
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def score_samples(self, X):
        pass
    

class ocnn(abstract_occ_model):
    def __init__(self,
                 input_dim,  # type: int
                 hidden_layer_size=32,  # type: int
                 batch_size=50,  # type: int
                 r=1.0,  # type: float
                 epochs=10,  # type: int
                 nu=0.10,  # type: float
                 ):
        # type: (...) -> None
        """
        :param input_dim: number of input features
        :param hidden_layer_size: number of neurons in the hidden layer
        :param r: bias of hyperplane
        """
        self.model = None
        self.history = None
        self.opt = None
        self.scores = None
        self.input_dim = input_dim
        self.hidden_size = hidden_layer_size
        self.r = r
        self.batch_size = batch_size 
        self.epochs = epochs
        self.nu = nu 
        self.build_model()
    
    def fit(self, X, init_lr=1e-4, save=False):
        # type: (ndarray, float, bool) -> None
        """
        builds and trains the model on the supplied input data
        :param X: input training data
        :param epochs: number of epochs to train for (default 50)
        :param nu: parameter between [0, 1] controls trade off between maximizing the distance of the hyperplane from
        the origin and the number of data points permitted to cross the hyper-plane (false positives) (default 1e-2)
        :param init_lr: initial learning rate (default 1e-2)
        :param save: flag indicating if the model should be  (default True)
        :return: trained model and callback history
        """
        
        self.opt = tf.optimizers.Adam(lr=init_lr, decay=init_lr / self.epochs)
        self.model.compile(optimizer=self.opt,
                      loss=self.loss, run_eagerly=True)

        # despite the fact that we don't have a ground truth `y`, the fit function requires a label argument,
        # so we just supply a dummy vector of 0s
        self.history = self.model.fit(X, np.zeros((X.shape[0],)),
                            batch_size=self.batch_size,
                            shuffle=True,
                            epochs=self.epochs,
                            verbose=1)
        
    def predict(self, X, log=True):
        # type: (ndarray, bool) -> ndarray
        if type(self.scores) != np.ndarray:
            self.score_samples(X)
        self.predictions = np.where(self.scores>=0, 1 ,np.where(self.scores<0, -1, self.scores))
        return self.predictions 
    
    def score_samples(self, X):
        """
        
        """
        y_hats = self.model.predict(X)
        r =  tfp.stats.percentile(tf.reduce_max(y_hats, axis=1), 100 * self.nu)
        print("r : {}".format(r))
        self.scores = (y_hats - r).numpy()
        return self.scores
    
    def build_model(self):
        # type: () -> None
        self.model = tf.keras.Sequential()
        self.input_layer = tfkl.Dense(self.hidden_size,
                                      use_bias=False,
                                      kernel_initializer="glorot_normal",
                                      kernel_regularizer=tfk.regularizers.l2(0.5),
                                      name="hidden_layer",
                                      activation=tf.nn.leaky_relu)
        self.layer_2 = tfkl.Dense(self.hidden_size,
                                      use_bias=False,
                                      kernel_initializer="glorot_normal",
                                      kernel_regularizer=tfk.regularizers.l2(0.5),
                                      name="hidden_layer",
                                      activation=tf.nn.leaky_relu)
        self.layer_3 = tfkl.Dense(self.hidden_size,
                                      use_bias=False,
                                      kernel_initializer="glorot_normal",
                                      kernel_regularizer=tfk.regularizers.l2(0.5),
                                      name="hidden_layer",
                                      activation=tf.nn.leaky_relu)

        # Define Dense layer from hidden to output
        self.output_layer = tfkl.Dense(1,
                                       use_bias=False,
                                       kernel_regularizer=tfk.regularizers.l2(0.5),
                                       name="hidden_output",
                                       )
        
        self.model.add(self.input_layer)
        self.model.add(self.output_layer)
        
    def loss(self, y_true, y_pred):
        # type: (EagerTensor, EagerTensor) -> EagerTensor
        y_pred_prev = y_pred
        loss_val = (1 / self.nu) * tf.keras.backend.mean(tf.keras.backend.maximum(0.0, self.r - y_pred), axis=-1)
        self.r = tfp.stats.percentile(tf.reduce_max(y_pred_prev, axis=1), 100 * self.nu)
        return loss_val
    
    
class ensemble(abstract_occ_model):
    """
    
    """
    def __init__(self, nu=0.1):
        self.base_estimators = [
            #OCSVM(contamination=nu),
            
            KNN(n_neighbors=100, contamination=nu),
            KNN(n_neighbors=25, contamination=nu),
            KNN(n_neighbors=5, contamination=nu),
            IForest(contamination=nu)
        ]
        self.model = SUOD(base_estimators=self.base_estimators,
                         rp_flag_global=False,
                         bps_flag=True,
                         approx_flag_global=False)
        self.scores = None
        
    def fit(self, X):
        self.model.fit(X)
        self.model.approximate(X)
        
    def predict(self,X):
        self.scores = self.compute_score(X)
        return np.where(self.scores>=0.5, 1 , np.where(self.scores<0.5, -1, self.scores))
    
    def score_samples(self,X):
        if type(self.scores) != np.ndarray:
            self.scores = self.compute_score(X)
            return self.scores
        else:
            return self.scores
    
    def compute_score(self,X):
        mean_prob = np.mean(self.model.predict_proba(X), axis=1)
        return mean_prob

    
class AutoEncoderOOD(abstract_occ_model):
    def __init__(self, hidden_neurons, nu, epochs):
        self.model = AutoEncoder(hidden_neurons=hidden_neurons, contamination=nu, epochs=epochs)
        
    def fit(self, X):
        self.model.fit(X)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score_samples(self, X):
        return -self.model.decision_function(X)
    
    
class VAE_ODD(AutoEncoderOOD):
    def __init__(self, hidden_neurons, nu, epochs):
        if len(hidden_neurons) % 2 == 0:
            print("The number of layers must be an odd number(2n+1).")
            sys.exit()
        encoder = hidden_neurons[0:len(hidden_neurons)//2]
        latent = hidden_neurons[len(hidden_neurons)//2]
        decoder = hidden_neurons[len(hidden_neurons)//2+1:len(hidden_neurons)]
        self.model = VAE(encoder_neurons=encoder,
                         decoder_neurons=decoder,
                         latent_dim=latent,
                         contamination=nu,
                         epochs=epochs
                        )