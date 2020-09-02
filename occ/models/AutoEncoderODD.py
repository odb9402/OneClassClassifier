from occ.models.abstract_occ_model import abstract_occ_model
import numpy as np
from pyod.models.auto_encoder import AutoEncoder

class AutoEncoderODD(abstract_occ_model):
    def __init__(self, hidden_neurons, nu, epochs, batch_size=32, output_activation='sigmoid'):
        self.model = AutoEncoder(hidden_neurons=hidden_neurons
            ,contamination=nu
            ,epochs=epochs
            ,batch_size=batch_size
            ,validation_size=0
            ,output_activation=output_activation)
        
    def fit(self, X):
        self.model.fit(X)
    
    def predict(self, X):
        prediction = self.model.predict(X)
        return np.where(prediction==0.0, 1 , np.where(prediction==1.0, -1, prediction))
    
    def score_samples(self, X):
        return -self.model.decision_function(X)