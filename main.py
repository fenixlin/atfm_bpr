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

    data = sp.csr_matrix(np.loadtxt(sys.argv[1]))
    attr = sp.csr_matrix(np.loadtxt(sys.argv[2]))
    num_folds = 5

    splitter = ds.DataSplitter(data, attr, num_folds)
    datamats = splitter.split_data()
    attrmats = splitter.split_attr()

    bpr_args = BPRArgs(0.1, 1.0, 0.2125, 0.0355, 0.0355)
    bpr_k = 3
    cv_iters = 30
    cv_folds = 4
    num_iters = 100
    avg_prec = 0
    avg_auc = 0

    assert num_folds>1
    assert cv_folds>1
    #training & testing
    for i in range(num_folds):
        tmp_data = copy(datamats)
        tmp_data.pop(i)
        tmp_attr = copy(attrmats)
        tmp_attr.pop(i)

        cv_parameter_set = [(0.03,0.03), (0.03,0.1), (0.1,0.03), (0.1,0.1)]
        model = mapper.Map_BPR(sp.hstack(tmp_data).tocsr(), sp.vstack(tmp_attr).tocsr(), bpr_k, bpr_args)

        #cv_parameter_set = [(0.03,0.03), (0.03,0.1), (0.1,0.03), (0.1,0.1)]
        #model = mapper.Map_Linear(sp.hstack(tmp_data).tocsr(), sp.vstack(tmp_attr).tocsr(), bpr_k, bpr_args)

        #cv_parameter_set = [1, 2, 3]
        #model = mapper.Map_KNN(sp.hstack(tmp_data).tocsr(), sp.vstack(tmp_attr).tocsr(), bpr_k, bpr_args)

        #model = mapper.CBF_KNN(sp.hstack(tmp_data).tocsr(), sp.vstack(tmp_attr).tocsr(), bpr_k, bpr_args)

        #model = mapper.Map_Random(sp.hstack(tmp_data).tocsr(), sp.vstack(tmp_attr).tocsr(), bpr_k, bpr_args)

        para = model.cross_validation(cv_iters, cv_parameter_set, cv_folds)
        model.set_parameter(para)
        model.train(num_iters)

        prec, auc = model.test(datamats[i].tocsr(), attrmats[i].tocsr())
        print "Test for fold",i,": Prec@n =",prec,"auc =",auc
        print "------------------------------------------------"
        avg_prec += prec
        avg_auc += auc
    print "avg_prec = ", avg_prec/num_folds, ", avg_auc = ", avg_auc/num_folds

