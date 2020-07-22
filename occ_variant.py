from occ import *
from occ_diff import *

class occ_variant(occ):
    def __init__(self):
        super().__init__()
        self.raw_X = None
    
    def load_data_csv(self, file_name, Y=False, **kwargs):
        super().load_data_csv(file_name, Y=Y, **kwargs)
        self.raw_X = self.X
        mean = np.mean(self.raw_X, axis=1).reshape([len(self.X),1])
        self.X = self.X - mean
        
    def export_score(self, file_name, **kwargs):
        if type(self.Y) != np.ndarray:
            if self.Y == None:
                array = np.concatenate((self.raw_X, self.get_score(**kwargs)), axis=1)
        array = np.concatenate((self.raw_X, self.Y, self.get_score(**kwargs)), axis=1)
        pd.DataFrame(array).to_csv(file_name, header=None, index=False)