"""
Main program to perform training, with a splitter to do k-fold cross-validation

"""
import mapper
import data_splitter as ds
from bpr import BPRArgs, BPR
from copy import copy
import numpy as np
import scipy.sparse as sp
import sys

if __name__ == '__main__':
    # example of training and testing with mapping functions
    if (len(sys.argv)!=3):
        print "example: python ./main.py __data_file__ __attr_file__"
        exit()

    data = sp.csc_matrix(np.loadtxt(sys.argv[1]))
    attr = sp.csr_matrix(np.loadtxt(sys.argv[2]))
    num_folds = 5

    splitter = ds.DataSplitter(data, attr, num_folds)
    datamats = splitter.split_data()
    attrmats = splitter.split_attr()

    bpr_args = BPRArgs(0.01, 1.0, 0.02125, 0.00355, 0.00355)
    bpr_k = 32
    num_iters = 3
    avg_prec = avg_auc = 0
    #training & testing
    for i in range(num_folds):
        tmp_data = copy(datamats)
        tmp_data.pop(i)
        tmp_attr = copy(attrmats)
        tmp_attr.pop(i)
        model = mapper.CBF_KNN(sp.hstack(tmp_data).tocsr(), sp.vstack(tmp_attr), bpr_k, bpr_args)
        #model = mapper.Map_Random(sp.hstack(tmp_data).tocsr(), sp.vstack(tmp_attr), bpr_k, bpr_args)
        model.train(num_iters)

        prec, auc = model.test(datamats[i].tocsr(), attrmats[i])
        print "Test for fold",i,": Prec@n =",prec,"auc =",auc
        print "------------------------------------------------"
        avg_prec += prec
        avg_auc += auc
    print "avg_prec = ", avg_prec/num_folds, ", avg_auc = ", avg_auc/num_folds

