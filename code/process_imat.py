import conv
import numpy as np

class Marshmallow():

    def __init__(self, path, format, id):
      

        self.path = path
        self.f = format
        
        self.imat = self._read_imzml()
        self.id = id

    def _read_imzml(self):
        imat_conv = conv.imat_conv(self.path, self.f)
        s = imat_conv.imat
        return s
    
    def norm(self, type):
        if type == 0: #tic norm
            
            self.imat =  self.imat.tic_normalise()
        if type == 1: #log norm
            self.imat = self.imat.log_transform()
        if type == 2:
            self.imat = self.imat.tic_normalise().log_transform()

    def segment(self, n_pc, n_clusters):

        self.imat.run_pca(n_pc)
        self.imat.run_kmeans(n_clusters=n_clusters)

    