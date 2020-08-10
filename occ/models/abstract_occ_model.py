from abc import *

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
