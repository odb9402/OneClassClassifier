from occ.models.abstract_occ_model import abstract_occ_model
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability as tfp
from tensorflow.python.framework.ops import EagerTensor
import numpy as np

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
                                      kernel_regularizer=tfk.regularizers.l2(0.2),
                                      name="hidden_layer",
                                      activation=tf.nn.leaky_relu)
        self.layer_2 = tfkl.Dense(self.hidden_size,
                                      use_bias=False,
                                      kernel_initializer="glorot_normal",
                                      kernel_regularizer=tfk.regularizers.l2(0.2),
                                      name="hidden_layer",
                                      activation=tf.nn.leaky_relu)
        self.layer_3 = tfkl.Dense(self.hidden_size,
                                      use_bias=False,
                                      kernel_initializer="glorot_normal",
                                      kernel_regularizer=tfk.regularizers.l2(0.2),
                                      name="hidden_layer",
                                      activation=tf.nn.leaky_relu)

        # Define Dense layer from hidden to output
        self.output_layer = tfkl.Dense(1,
                                       use_bias=False,
                                       kernel_regularizer=tfk.regularizers.l2(0.2),
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