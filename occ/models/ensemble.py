from occ.models.abstract_occ_model import abstract_occ_model
import numpy as np
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.lscp import LSCP
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE 
from suod.models.base import SUOD

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
