import scipy.sparse as sp
import sys
import random

class DataSplitter(object):

    def __init__(self, datamat, attrmat, k):
        assert sp.isspmatrix_csr(datamat)
        assert sp.isspmatrix_csr(attrmat)
        self.datamat = datamat.tocsc()
        self.attrmat = attrmat
        self.k = k
        _, self.num_items = datamat.shape
        assert self.k<=self.num_items
        self.index = [i for i in range(self.num_items)]
        #random.shuffle(self.index)

    def split_data(self):
        base = 0
        result = []
        for i in range(self.k):
            tmp = []
            for j in range(min(self.num_items-base, self.num_items/self.k)):
                tmp.append(self.datamat.getcol(self.index[base+j]))
            base += self.num_items/self.k
            result.append(sp.hstack(tmp))
        return result

    def split_attr(self):
        base = 0
        result = []
        for i in range(self.k):
            tmp = []
            for j in range(min(self.num_items-base, self.num_items/self.k)):
                tmp.append(self.attrmat.getrow(self.index[base+j]))
            base += self.num_items/self.k
            result.append(sp.vstack(tmp))
        return result

