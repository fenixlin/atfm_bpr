"""
Attribute to feature mapping functions for items

map-knn(k) : predict matrix factor H with cosine similarity as weight.
map-lin(alpha, lambda) : linear model train with LeastMSE cost function. 
map-bpr(alpha, lambda) : linear model train with BPR-OPT cost function.
cbf-knn() : k=infinity, predict score directly with sum of cosine similarity.
random() : predict a random set of items as baseline

Inside brackets are parameters needed which can be train via cross-validation.

A, W, H, S matrix might be needed.

"""

import bpr
from math import sqrt
import numpy as np
import scipy.sparse as sp
import data_splitter as ds
from copy import copy
import sys
import random

class Mapper(object):

    def __init__(self):
        pass

    def init(self, data, attr, bpr_k=None, bpr_args=None):
        assert sp.isspmatrix_csr(data)
        assert sp.isspmatrix_csr(attr)
        self.data = data
        self.num_users, self.num_items = data.shape
        print "Mapper train data : ", self.num_users, "x", self.num_items
        self.attr = attr
        assert attr.shape[0] >= self.num_items
        self.bpr_k = [self.num_users/5,bpr_k][bpr_k!=None]
        if bpr_args==None:
            self.bpr_args = bpr.BPRArgs(0.01, 1.0, 0.02125, 0.00355, 0.00355)
        else:
            self.bpr_args = bpr_args
        self.bpr_model = bpr.BPR(self.bpr_k, self.bpr_args)
        sample_negative_items_empirically = False
        self.sampler = bpr.UniformUserUniformItem(sample_negative_items_empirically)
    
    def test_init(self, test_data, test_attr):
        assert sp.isspmatrix_csr(test_data)
        assert sp.isspmatrix_csr(test_attr)
        self.num_test_items, _ = test_attr.shape
        self.test_data = test_data
        self.test_attr = test_attr

    def _row_dot(self, x, y):
        return x.dot(y.transpose()).sum()

    def cos_similarity(self, i):
        try:
            assert len(self.attr_sqr_cache) == len(self.attr)
            assert len(self.tattr_sqr_cache) == len(self.test_attr)
        except:
            self.attr_sqr_cache = []
            self.tattr_sqr_cache = []
            for j in range(self.num_items):
                self.attr_sqr_cache.append(self._row_dot(self.attr[j], self.attr[j]))
            for j in range(self.num_test_items):
                self.tattr_sqr_cache.append(self._row_dot(self.test_attr[j], self.test_attr[j]))
        similarity = []
        for j in range(self.num_items):
            similarity.append(self._row_dot(self.test_attr[i], self.attr[j]) / sqrt(self.tattr_sqr_cache[i] * self.attr_sqr_cache[j]))
        return similarity

    def accuracy(self, threshold=0.5):
        result = 0.0
        for i in range(self.num_items):
            pred_i = self.map_predict(i)
            for u in range(self.num_users):
                posidx = self.test_data[u].indices
                if (pred_i[u]>=threshold and (i in posidx)) or (pred_i[u]<threshold and (not i in posidx)):
                    result +=1
        result /= (self.num_items * self.num_users)
        return result

    def prec_at_n(self, prec_n):
        #precision of top-n recommended results, average across users
        prec_n = min(prec_n, self.num_test_items)
        result = 0
        
        cand = [[] for i in range(self.num_users)]
        for i in range(self.num_test_items):
            pred_i = self.map_predict(i)
            for u in range(self.num_users):
                cand[u].append((pred_i[u], i))
        for u in range(self.num_users):
            cand[u].sort(lambda x,y : cmp(x[0],y[0]), reverse=True)
            tmp = 0.0
            row_u = self.test_data[u].toarray()[0]
            for i in range(prec_n):
                if row_u[cand[u][i][1]]>0:
                    tmp += 1
            result += tmp/prec_n
        result /= self.num_users
        return result

    def auc(self):
        #area under ROC curve, compute , average across users
        result = 0
        pred = [[] for i in range(self.num_users)]
        for i in range(self.num_test_items):
            self._cache_attr(i)
            for u in range(self.num_users):
                pred[u].append(self.map_predict(u, i, caching))
        for u in range(self.num_users):
            tmp = 0.0
            posidx = self.test_data[u].indices
            for j in range(self.num_test_items):
                if j in posidx:
                    continue
                for i in posidx:
                    if pred[u][i]-pred[u][j]>=0:
                        tmp += 1
            real_pos = len(self.test_data[u].indices)
            result += tmp/max(real_pos, 1)/max(self.num_test_items-real_pos, 1)
        result /= self.num_users
        return result 

    def cross_validation(self, cv_num_iters, cv_set, cv_fold):
        origin_data = self.data
        origin_attr = self.attr
        splitter = ds.DataSplitter(origin_data, origin_attr, cv_fold)
        datamats = splitter.split_data()
        attrmats = splitter.split_attr()
        bestacc = 0
        bestpara = None
        for para in cv_set:
            self.set_parameter(para)
            avg_acc = 0
            for i in range(cv_fold):
                tmp_data = copy(datamats)
                tmp_data.pop(i)
                tmp_attr = copy(attrmats)
                tmp_attr.pop(i)
                self.init(tmp_data, tmp_attr, self.bpr_k, self.bpr_args)
                self.train(cv_num_iters)
                self.test_init(datamats[i])
                avg_acc += self.accuracy()
            avg_acc /= cv_fold
            if (avg_acc > bestacc):
                bestpara = para
        self.data = origin_data
        self.attr = origin_attr
        return para

class Map_KNN(Mapper):

    def __init__(self, data, attr, bpr_k=None, bpr_args=None, k=1):
        self.init(data, attr, bpr_k, bpr_args)
        self.k = k

    def set_parameter(self, k):
        self.k = k

    def train(self, num_iters):
        self.bpr_model.train(self.data, self.sampler, num_iters)
        
    def test(self, test_data, test_attr, prec_n=5):
        self.test_init(test_data, test_attr)
        return [self.prec_at_n(prec_n), self.auc()]   

    def map_predict(self, i):
        result = []
        cos_sim = self.cos_similarity(i)
        cand = [(cos_sim[i], i) for i in range(self.num_users)]
        cand.sort(lambda x,y: cmp(x[0],y[0]), reverse=True)
        #average new h from top-k h vectors, and predict with bpr
        i_factors = np.zeros(self.bpr_k)
        i_bias = 0
        for j in range(self.k):
            i_factors += self.bpr_model.item_factors[cand[j][1],:]
            i_bias += self.bpr_model.item_bias[cand[j][1]]
        i_factors /= self.k
        i_bias /= self.k
        for j in range(self.num_users):
            result.append(i_bias + np.dot(self.bpr_model.user_factors[u], i_factors))
        return result

class Map_Linear(Mapper):

    def __init__(self, data, attr, bpr_k=None, bpr_args=None, learning_rate=None, penalty_factor=None):
        self.init(data, attr, bpr_k, bpr_args)
        self.learning_rate = learning_rate
        self.penalty_factor = penalty_factor

    def train(self, num_iters):
        #TODO
        #train a linear model across attributes(X=attrs, Y=H)
        pass

    def test(self, test_data, test_attr, prec_n=5):
        #TODO
        pass

    def set_parameter(self, para_set):
        self.learning_rate = para_set[0]
        self.penalty_factor = para_set[1]

class Map_BPR(Mapper):

    def __init__(self, data, attr, bpr_k=None, bpr_args=None, learning_rate=None, penalty_factor=None):
        self.init(data, attr, bpr_k, bpr_args)
        self.learning_rate = learning_rate
        self.penalty_factor = penalty_factor

    def train(self, num_iters):
        #TODO
        #train a linear model across attributes(X=attrs, Y=H)
        pass

    def test(self, test_data, test_attr, prec_n=5):
        #TODO
        pass

    def set_parameter(self, para_set):
        self.learning_rate = para_set[0]
        self.penalty_factor = para_set[1]

class CBF_KNN(Mapper):

    def __init__(self, data, attr, bpr_k=None, bpr_args=None, k=None):
        self.init(data, attr, bpr_k, bpr_args)
        self.k = k

    def set_parameter(self, k):
        self.k = k

    def train(self, num_iters):
        # underlying bpr model is useless, so no need to train
        pass 

    def test(self, test_data, test_attr, prec_n=5):
        self.test_init(test_data, test_attr)
        return [self.prec_at_n(prec_n), self.auc()]   

    def map_predict(self, i):
        result = []
        cos_sim = self.cos_similarity(i)
        if self.k==None:
            # k is infinity by default
            for u in range(self.num_users):
                pred_j = 0
                for j in self.data[u].indices:
                    pred_j += cos_sim[j]
                result.append(pred_j)
        else:
            for u in range(self.num_users):
                cand = []
                for j in self.data[u].indices:
                    cand.append(cos_sim[j])
                cand.sort()
                pred_j = 0
                for j in range(self.k):
                    pred_j += cand[j]
                result.append(pred_j)
        return result

class Map_Random(Mapper):

    def __init__(self, data, attr, bpr_k=None, bpr_args=None):
        self.init(data, attr, bpr_k, bpr_args)

    def train(self, num_iters):
        #no need to train
        pass

    def test(self, test_data, test_attr, prec_n=5):
        self.test_init(test_data, test_attr)
        return [self.prec_at_n(prec_n), self.auc()]   

    def map_predict(self, i, max_score=1.0):
        return [(random.random() * max_score) for i in range(self.num_users)]
