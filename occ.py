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
from scipy import io
from sklearn.model_selection import train_test_split as train_test_split
from sklearn.decomposition import PCA as PCA
from sklearn.decomposition import KernelPCA as kPCA
from sklearn.preprocessing import Normalizer
from sklearn.base import BaseEstimator, ClassifierMixin
matplotlib.style.use("ggplot")
#tf.__version__
tf.test.is_gpu_available()
#gpus = tf.config.experimental.list_physical_devices('GPU')

class occ():
    """
    
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
        epoches = 10
        batch_size = 50
        nu = 0.05
        
        for k, v in kwargs.items():
            if 'kernel' == k:
                kernel_set = v
            if 'gamma' == k:
                gamma_set = v
            if 'epoches' == k:
                epoches = v
            if 'nu' == k:
                nu = v
            if 'batch_size' == k:
                batch_size = v
                
        if model=='ocsvm':
            self.model = sklearn.svm.OneClassSVM(gamma=gamma_set, kernel=kernel_set, nu=nu)
        elif model=='ocnn':
            self.model = ocnn(len(data[0]), epoches=epoches, nu=nu, batch_size=batch_size)
        else:
            print("There is no such model type {}".format(model))
        
        data = self.select_data(data, **kwargs)

        self.model.fit(data)
    
    def predict(self, data=None, **kwargs):
        if type(data) != np.ndarray:
            if data == None:
                data = self.X
        data = self.select_data(data, **kwargs)
        return self.model.predict(data)
    
    def get_score(self, data=None, **kwargs):
        if type(data) != np.ndarray:
            if data == None:
                data = self.X
        data = self.select_data(data, **kwargs)
        return self.model.score_samples(data).reshape(len(data),1)
    
    def export_score(self, file_name, **kwargs):
        if type(self.Y) != np.ndarray:
            if self.Y == None:
                array = np.concatenate((self.X, self.get_score(**kwargs)), axis=1)
        array = np.concatenate((self.X, self.Y, self.get_score(**kwargs)), axis=1)
        pd.DataFrame(array).to_csv(file_name, header=None, index=False)
    
    @staticmethod
    def select_data(data, **kwargs):
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


class ocnn():
    def __init__(self, input_dim, hidden_layer_size=64, batch_size=50, r=1.0, epoches=10, nu=0.10):
        """
        :param input_dim: number of input features
        :param hidden_layer_size: number of neurons in the hidden layer
        :param r: bias of hyperplane
        """
        self.model = None
        self.history = None
        self.opt = None
        self.input_dim = input_dim
        self.hidden_size = hidden_layer_size
        self.r = r
        self.batch_size = batch_size 
        self.epoches = epoches
        self.nu = nu 
        self.build_model()
    
    def fit(self, X, init_lr=1e-4, save=False):
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
        
        self.opt = tf.optimizers.Adam(lr=init_lr, decay=init_lr / self.epoches)
        self.model.compile(optimizer=self.opt,
                      loss=self.loss, run_eagerly=True)

        # despite the fact that we don't have a ground truth `y`, the fit function requires a label argument,
        # so we just supply a dummy vector of 0s
        self.history = self.model.fit(X, np.zeros((X.shape[0],)),
                            batch_size=self.batch_size,
                            shuffle=True,
                            epochs=self.epoches,
                            verbose=1)
        
    def predict(self, X, log=True):
        y_hats = self.model.predict(X)
        r =  tfp.stats.percentile(tf.reduce_max(y_hats, axis=1), 100 * self.nu)
        print("r : {}".format(r))
        self.predictions = (y_hats - r).numpy()
        return self.predictions 
    
    def score_samples(self, X):
        """
        
        """
        if type(self.predictions) == np.ndarray:
            self.predict(X)
        return self.predictions
    
    def build_model(self):
        self.model = tf.keras.Sequential()
        self.input_layer = tfkl.Dense(self.hidden_size,
                                      use_bias=False,
                                      kernel_initializer="glorot_normal",
                                      kernel_regularizer=tfk.regularizers.l2(0.5),
                                      name="hidden_layer")#,
                                      #activation=tf.nn.leaky_relu)

        # Define Dense layer from hidden to output
        self.output_layer = tfkl.Dense(1,
                                       use_bias=False,
                                       kernel_regularizer=tfk.regularizers.l2(0.5),
                                       name="hidden_output",
                                       )
        
        self.model.add(self.input_layer)
        self.model.add(self.output_layer)
        
    def loss(self, y_true, y_pred):
        y_pred_prev = y_pred
        loss_val = (1 / self.nu) * tf.keras.backend.mean(tf.keras.backend.maximum(0.0, self.r - y_pred), axis=-1)
        self.r = tfp.stats.percentile(tf.reduce_max(y_pred_prev, axis=1), 100 * self.nu)
        return loss_val