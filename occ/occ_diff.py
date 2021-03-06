from occ.occ import *

class occ_diff(occ):
    def __init__(self):
        super().__init__()
        self.raw_X = None
    
    def load_data_csv(self, file_name, Y=False, **kwargs):
        super().load_data_csv(file_name, Y=Y, **kwargs)
        self.raw_X = self.X
        self.X = np.diff(self.X)
        
    def export_score(self, file_name, **kwargs):
        if type(self.Y) != np.ndarray:
            if self.Y == None:
                array = np.concatenate((self.raw_X, self.get_score(**kwargs)), axis=1)
        array = np.concatenate((self.raw_X, self.Y, self.get_score(**kwargs)), axis=1)
        pd.DataFrame(array).to_csv(file_name, header=None, index=False)