"""
Main program to perform training, with a splitter to do k-fold cross-validation

"""
import mapper
from bpr import BPRArgs, BPR
import numpy as np
import scipy.sparse as sp
import sys
import random

class DataSplitter(object):

    def __init__(self, datamat, attrmat, k):
        assert sp.isspmatrix_csc(datamat)
        assert sp.isspmatrix_csr(attrmat)
        self.datamat = datamat
        self.attrmat = attrmat
        self.k = k
        _, self.num_items = datamat.shape
        self.index = [i for i in range(self.num_items)]
        random.shuffle(self.index)

    def split_data(self):
        base = 0
        result = []
        for i in range(self.k):
            tmp = []
            for j in range(self.num_items/self.k):
                tmp.append(self.datamat.getcol(self.index[base+j]))
            base += self.num_items/self.k
            result.append(sp.hstack(tmp))
        return result

    def split_attr(self):
        base = 0
        result = []
        for i in range(self.k):
            tmp = []
            for j in range(self.num_items/self.k):
                tmp.append(self.attrmat.getrow(self.index[base+j]))
            base += self.num_items/self.k
            result.append(sp.vstack(tmp))
        return result

if __name__ == '__main__':
    # example of training and testing with mapping functions
    if (len(sys.argv)!=3):
        print "example: python ./main.py __data_file__ __attr_file__"
        exit()

    data = sp.csc_matrix(np.loadtxt(sys.argv[1]))
    attr = sp.csr_matrix(np.loadtxt(sys.argv[2]))
    num_folds = 5

    splitter = DataSplitter(data, attr, num_folds)
    datamats = splitter.split_data()
    attrmats = splitter.split_attr()

    bpr_args = BPRArgs(0.01, 1.0, 0.02125, 0.00355, 0.00355)
    bpr_k = 32
    num_iters = 3
    avg_prec = avg_auc = 0
    #training & testing
    for i in range(num_folds):
        tmp_data = datamats
        del tmp_data[i]
        tmp_attr = attrmats
        del tmp_attr[i]
        model = mapper.CBF_KNN(sp.hstack(tmp_data).tocsr(), sp.vstack(tmp_attr), bpr_k, bpr_args)
        model.train(num_iters)

        prec, auc = model.test(datamats[i].tocsr(), attrmats[i])
        print "Test for fold",i,": Prec@n =",prec,"auc =",auc
        print "------------------------------------------------"
        avg_prec += prec
        avg_auc += auc
    print "avg_prec = ", avg_prec/num_folds, ", avg_auc = ", avg_auc/num_fold

