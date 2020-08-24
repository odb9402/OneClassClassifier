import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow.keras as tfk
import numpy as np
import sys
import datetime
import pydot
import graphviz
from sklearn.model_selection import train_test_split

from occ.models.abstract_occ_model import abstract_occ_model

class deep_SVDD(abstract_occ_model):
    """
    Tensorflow implementation of Deep-SVDD, the semi-supervised and
    unsupervised outlier detection algorithm. If we use the semi-supervised condition,
    the objective function to be optimized will be different.
    
    <ICML2018> Deep one-class classification; Ruff, Lukas et al.
    
    Attributes:
        X
        lr
        kernel
        radius
        center
        dist
        objective
    
    Methods:
        
    
    """
    def __init__(self, nu=0.1, known_normal=False, hidden_neurons=None,
                 kernel_epochs=50, radius_epochs=100, batch_size=50):
        if hidden_neurons == None:
            self.hidden_neurons = [32,64,64,64]
        else:
            self.hidden_neurons = hidden_neurons
        self.kernel_epochs = kernel_epochs
        self.radius_epochs = radius_epochs
        self.batch_size = batch_size
        self.nu = nu
        self.kernel_lr = tfk.optimizers.schedules.ExponentialDecay(0.0001,
                                                                  decay_steps=1000,
                                                                  decay_rate=0.98) 
        self.radius_lr = tfk.optimizers.schedules.ExponentialDecay(0.01,
                                                                  decay_steps=10000,
                                                                  decay_rate=0.96) 
        self.center_offset = 0.
        self.optimizer = tfk.optimizers.Adam(learning_rate=self.kernel_lr)
        self.known_normal = known_normal
        
    def fit(self, X, y=None, debug=False):
        """
        Since the deep_svdd is a semi-supervised algorithm assuming that we only have
        positive(normal) samples, X should be known normal samples.
        
        """
        self.train_X, self.test_X = train_test_split(X, test_size=0.2)
        self.features = X.shape[-1]
        self.kernel, self.__radius_layer = self.__build_deep_kernel_dense(self.hidden_neurons)
        
        if debug:
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        else:
            tensorboard_callback = None
            
        print("@ Kernel parameters learning")
        # Freeze the hypersphere radius
        self.__radius_layer.trainable = False
        self.kernel.compile(optimizer=self.optimizer)
        self.history = self.kernel.fit(self.train_X, None,
                                    validation_data=(self.test_X, None),
                                    epochs=self.kernel_epochs,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    verbose=0)
                                    #callbacks=[tensorboard_callback]).history
        # Freeze kernel layers
        for i in range(1,len(self.hidden_neurons)+1):
            self.kernel.layers[i].trainable = False
            self.kernel.layers[i].kernel_regularizer = None
        self.__radius_layer.trainable = True
        self.kernel.add_loss(lambda: tf.math.square(self.__radius_layer.rad)) ## Second loss: radius loss
        self.optimizer = tfk.optimizers.Adam(learning_rate=self.radius_lr)
        self.kernel.compile(optimizer=self.optimizer)
        
        print("@ Radius learning (Freeze kernel parameters)")
        self.history_radius = self.kernel.fit(self.train_X, None,
                                             validation_data=(self.test_X, None),
                                             epochs=self.radius_epochs + self.kernel_epochs,
                                             initial_epoch=self.kernel_epochs,
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             verbose=0)
                                             #callbacks=[tensorboard_callback]).history
        self.radius = self.__radius_layer.get_weights()[0]
        print("@ Radius of the Hypersphere : {}".format(self.radius))
    
    def predict(self, X):
        center_dist = -self.score_samples(X)
        r = self.radius*self.radius
        predictions = np.where(center_dist < r, 1, -1)
        return predictions
        
    def score_samples(self, X):
        center = np.mean(self.kernel.predict(self.train_X), axis=0) + self.center_offset
        #print("The center : {}".format(center))
        dist = self.kernel.predict(X) - center
        dist = np.sqrt(np.sum(np.square(dist), axis=1))
        return -dist
    
    def __build_deep_kernel_dense(self, hidden_neurons, batch_norm=False):
        """
        High-dimensional non-linear mapping for input data.
        Conditions :
            1. Bias terms should not be used.
            2. Unbunded activations or only zero bounded activations should be used
              eg) ReLU
            3. The weight matrices should not be zero
            
        return:
            A. (model): The distances from the hypersphere boundary.
                It can be minus if the data is inside of the boundary.
            
            B. (radian distance_layer): The layer including the hypersphere radius.
            
        """
        input_layer = tfk.Input(shape=(self.features))
        
        regularizer = lambda W: self.__zero_block_regularizer(W, add_non_zero=False, add_l2=True
                                                              ,non_zero_weight= 0.001, l2_weight=0.001)
        
        kernel_layers = []
        func = tfkl.Dense(hidden_neurons[0], activation=None,
                                     kernel_regularizer=regularizer, use_bias=False,
                                     kernel_initializer='glorot_uniform')(input_layer)
        if batch_norm:
            func = tfkl.BatchNormalization(center=False)(func)
        func = tfkl.Activation('relu')(func)
        kernel_layers.append(func)
        
        for i in range(0,len(hidden_neurons)-1):
            if(i == len(hidden_neurons)-2):
                #Last layer
                kernel_layers.append(tfkl.Dense(hidden_neurons[i+1], activation=None,
                           kernel_regularizer=regularizer, use_bias=False,
                           kernel_initializer='glorot_uniform', name='kernel_output')(kernel_layers[i]))
            else:
                func = tfkl.Dense(hidden_neurons[i+1], activation=None,
                           kernel_regularizer=regularizer, use_bias=False,
                           kernel_initializer='glorot_uniform')(kernel_layers[i])
                if batch_norm:
                    func = tfkl.BatchNormalization(center=False)(func)
                func = tfkl.Activation('relu')(func)
                kernel_layers.append(func)
        
        if self.known_normal:
            model = tfk.Model(
                inputs = input_layer,
                outputs = kernel_layers[-1])
        else:
            radian_distance_layer = HypersphereDistanceLayer(self.center_offset, name='hypersphere_distance')
            radian_distance = radian_distance_layer(kernel_layers[-1])
            
            model = tfk.Model(
                inputs = input_layer,
                outputs = kernel_layers[-1])
        
        objective = lambda y_true, y_pred :self.__build_penalty_objective(y_pred,
                                                                  self.nu,
                                                                  self.known_normal)
        model.add_loss(objective(None, radian_distance)) ## First loss : radius distance
        model.compile(optimizer=self.optimizer)
        print(model.summary())
        #tfk.utils.plot_model(model,"model.png", show_shapes=True)
        return model, radian_distance_layer
    
    def __zero_block_regularizer(self, weights, add_non_zero=True, add_l2=True
                                 ,non_zero_weight=0.1, l2_weight=0.1):
        """
        Experiments: what if there is a regularizer to prevent all weights are going to
            collapse into 0?
            by Dongpin Oh
            
            well suited!
        """
        reg = 0. 
        if add_non_zero:
            reg = reg + tf.math.divide(self.non_zero_regularizer,tf.norm(weights, ord=2))
        if add_l2:
            reg = reg + l2_weight*tf.norm(weights, ord=2)
        return reg
            
    def __build_deep_kernel_convolve(self):
        model = tfk.models.Sequential()
        return model
    
    def __build_penalty_objective(self, X, nu, known_normal=False):
        """
        Unsupervised:
        argmin_R,W = R^2 + mean(max(0, ||nn(X,W) - C||^2 - R^2))/nu + regularizer
        
        Semi-supervised:
        argmin_R,W = mean(max(0, ||nn(X,W) - C||^2 - R^2)) + regularizer
        
        penalty term: ||nn(X,W) - C||^2 - R^2
        
        :param known_normal: (boolean)
            nu: (float, [0-1])
        
        """
        #C = tf.math.reduce_mean(X, axis=0)
        #dist = tf.math.reduce_sum(tf.math.square(X-C), axis=1) 
        
        if known_normal:
            penalty = tf.math.reduce_mean(X)
            return penalty
        elif not known_normal:
            penalty = tf.math.maximum(0.0, X)
            penalty = tf.math.divide(tf.math.reduce_mean(penalty), nu)
            return penalty
        else:
            sys.exit("error")

            
class deep_kernel_model(tfk.Model):
    def __init__(self, r=None, **kwargs):
        super(deep_kernel_model, self).__init__(**kwargs)
        
        
class HypersphereDistanceLayer(tfkl.Layer):
    def __init__(self, center_offset, **kwargs):
        super(HypersphereDistanceLayer, self).__init__(**kwargs)
        self.center_offset = center_offset
    
    def build(self, input_shape):
        self.rad = tf.Variable(10., trainable=True, name='rad')
        
    def call(self, inputs):
        """
        f(inputs, rads) = ||Inputs - center(inputs)||^2 - rads^2
        
        """
        #self.add_loss(tf.math.square(self.rad)) ## Regularizer(Reward) for the hypersphere radius
        
        C = tf.math.reduce_mean(inputs, axis=0, name='center') + self.center_offset
        dist = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(inputs-C), axis=1), name='distance')
        radian_distance = tf.math.subtract(dist, tf.math.square(self.rad), name='radian_distance')
        
        return radian_distance

    
class CenterDistanceLayer(tfkl.Layer):
    def __init__(self, **kwargs):
        super(CenterDistanceLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        C = tf.math.reduce_mean(inputs, axis=0, name='center')
        dist = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(inputs-C), axis=1), name='distance')
        return dist