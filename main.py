"""
Copyright (c) 2015 Fenix Lin

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

    #all parameters needed setting are here
    model_id = 4 # 0=Map_BPR 1=Map_Linear 2=Map_KNN 3=CBF_KNN 4=Random
    num_folds = 4
    bpr_args = BPRArgs(0.01, 1.0, 0.02125, 0.00355, 0.00355)
    bpr_k = 24
    cv_iters = 10
    cv_folds = 3
    num_iters = 20

    data = sp.csc_matrix(np.loadtxt(sys.argv[1]))
    attr = np.loadtxt(sys.argv[2])
    splitter = ds.DataSplitter(data, attr, num_folds)
    datamats = splitter.split_data()
    attrmats = splitter.split_attr()

    assert num_folds>1
    assert cv_folds>1
    avg_prec = 0
    avg_auc = 0
    #training & testing
    for i in range(num_folds):
        tmp_data = copy(datamats)
        tmp_data.pop(i)
        tmp_attr = copy(attrmats)
        tmp_attr.pop(i)

        if (model_id == 0):
            cv_parameter_set = [(0.03,0.03), (0.03,0.1), (0.1,0.03), (0.1,0.1)]
            model = mapper.Map_BPR(sp.hstack(tmp_data,"csc"), np.vstack(tmp_attr), bpr_k, bpr_args)
        elif (model_id == 1):
            cv_parameter_set = [(0.03,0.03), (0.03,0.1), (0.1,0.03), (0.1,0.1)]
            model = mapper.Map_Linear(sp.hstack(tmp_data,"csc"), np.vstack(tmp_attr), bpr_k, bpr_args)
        elif (model_id == 2):
            cv_parameter_set = [1, 2, 3]
            model = mapper.Map_KNN(sp.hstack(tmp_data,"csc"), np.vstack(tmp_attr), bpr_k, bpr_args)
        elif (model_id == 3):
            model = mapper.CBF_KNN(sp.hstack(tmp_data,"csc"), np.vstack(tmp_attr), bpr_k, bpr_args)
        elif (model_id == 4):
            model = mapper.Map_Random(sp.hstack(tmp_data,"csc"), np.vstack(tmp_attr), bpr_k, bpr_args)

        if (model_id<3):
            para = model.cross_validation(cv_iters, cv_parameter_set, cv_folds)
            model.set_parameter(para)
        model.train(num_iters)

        prec, auc = model.test(datamats[i], attrmats[i])
        print "Test for fold",i,": Prec@n =",prec,"auc =",auc
        print "------------------------------------------------"
        avg_prec += prec
        avg_auc += auc
    print "avg_prec = ", avg_prec/num_folds, ", avg_auc = ", avg_auc/num_folds

