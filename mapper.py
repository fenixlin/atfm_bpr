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
from math import sqrt, exp
import numpy as np
import scipy.sparse as sp
import data_splitter as ds
from copy import copy
import sys
import random

class Mapper(object):

    def __init__(self):
        pass

    def init(self, data, attr, bpr_k=None, bpr_args=None, bpr_model=None):
        assert sp.isspmatrix_csr(data)
        self.data = data
        self.num_users, self.num_items = data.shape
        self.attr = attr
        assert attr.shape[0] >= self.num_items
        _, self.num_attrs = attr.shape
        if bpr_model==None:
            self.bpr_k = [self.num_users/5,bpr_k][bpr_k!=None]
            if bpr_args==None:
                self.bpr_args = bpr.BPRArgs(0.01, 1.0, 0.02125, 0.00355, 0.00355)
            else:
                self.bpr_args = bpr_args
            self.bpr_model = bpr.BPR(self.bpr_k, self.bpr_args)
        else:
            self.bpr_model = bpr_model
            self.bpr_k = bpr_model.D
            self.bpr_args = bpr.BPRArgs(bpr_model.learning_rate, \
                bpr_model.bias_regularization, \
                bpr_model.user_regularization, \
                bpr_model.positive_item_regularization, \
                bpr_model.negative_item_regularization, \
                bpr_model.update_negative_item_factors)
        self.sampler = bpr.UniformUserUniformItem()
    
    def test_init(self, test_data, test_attr):
        assert sp.isspmatrix_csr(test_data)
        self.num_test_items, _ = test_attr.shape
        self.test_data = test_data
        self.test_attr = test_attr

    def cos_similarity(self, i):
        try:
            assert len(self.attr_sqr_cache) == len(self.attr)
            assert len(self.tattr_sqr_cache) == len(self.test_attr)
        except:
            self.attr_sqr_cache = []
            self.tattr_sqr_cache = []
            for j in range(self.num_items):
                self.attr_sqr_cache.append(sqrt(np.dot(self.attr[j], self.attr[j])))
            for j in range(self.num_test_items):
                self.tattr_sqr_cache.append(sqrt(np.dot(self.test_attr[j], self.test_attr[j])))

        similarity = []
        for j in range(self.num_items):
            similarity.append(np.dot(self.test_attr[i], self.attr[j]) / (self.tattr_sqr_cache[i] * self.attr_sqr_cache[j]))
        return similarity

    def accuracy(self, threshold=0.5):
    #XXX: bpr models have no range bound, while its focus are pair-wise relationships, so it's hard to set a threshold and test accuracy
        result = 0.0
        for i in range(self.num_test_items):
            pred_i = self.test_predict(i)
            for u in range(self.num_users):
                posidx = self.test_data[u].indices
                if (pred_i[u]>=threshold and (i in posidx)) or (pred_i[u]<threshold and (not i in posidx)):
                    result += 1
        result /= (self.num_items * self.num_users)
        return result

    def prec_at_n(self, prec_n):
        #precision of top-n recommended results, average across users
        assert prec_n <= self.num_test_items
        result = 0
        
        cand = [[] for i in range(self.num_users)]
        for i in range(self.num_test_items):
            pred_i = self.test_predict(i)
            for u in range(self.num_users):
                cand[u].append((pred_i[u], i))
        for u in range(self.num_users):
            cand[u].sort(lambda x,y : cmp(x[0],y[0]), reverse=True)
            tmp = 0.0
            row_u = self.test_data[u].indices
            for i in range(prec_n):
                if cand[u][i][1] in row_u:
                    tmp += 1
            result += tmp/prec_n
        result /= self.num_users
        return result

    def auc(self):
        #area under ROC curve, compute , average across users
        result = 0
        pred = [[] for i in range(self.num_users)]
        for i in range(self.num_test_items):
            pred_i = self.test_predict(i)
            for u in range(self.num_users):
                pred[u].append(pred_i[u])
        for u in range(self.num_users):
            tmp = 0.0
            posidx = self.test_data[u].indices
            for j in range(self.num_test_items):
                if j in posidx:
                    continue
                for i in posidx:
                    if pred[u][i]-pred[u][j]>=0:
                        tmp += 1
            real_pos = len(posidx)
            result += tmp/max(real_pos, 1)/max(self.num_test_items-real_pos, 1)
        result /= self.num_users
        return result 

    def cross_validation(self, cv_num_iters, cv_set, cv_folds):
        origin_data = self.data
        origin_attr = self.attr
        origin_model = self.bpr_model
        splitter = ds.DataSplitter(origin_data, origin_attr, cv_folds)
        datamats = splitter.split_data()
        attrmats = splitter.split_attr()
        bestscore = 0.0
        bestpara = None
        for para in cv_set:
            self.set_parameter(para)
            avg_score = 0.0
            print "Cross-validating parameter",para,"........."
            for i in range(cv_folds):
                tmp_data = copy(datamats)
                tmp_data.pop(i)
                tmp_attr = copy(attrmats)
                tmp_attr.pop(i)
                self.init(sp.hstack(tmp_data).tocsr(), np.vstack(tmp_attr), self.bpr_k, self.bpr_args)
                self.train(cv_num_iters)
                self.test_init(datamats[i].tocsr(), attrmats[i])
                #avg_score += self.accuracy()
                cur_score = self.prec_at_n(5)
                print "prec@5 of cross-validation fold",i,":",cur_score
                avg_score += cur_score
            avg_score /= cv_folds
            print "Average score for parameter after cross-validation",para,":",avg_score
            if (avg_score > bestscore):
                bestpara = para
                bestscore = avg_score
        #print "best parameter in cross-validation :", bestpara, "with accuracy", bestscore
        print "best parameter in cross-validation :", bestpara, "with prec@n", bestscore
        self.init(origin_data, origin_attr, None, None, origin_model)
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

    def test_predict(self, i):
        result = []
        cos_sim = self.cos_similarity(i)
        cand = [(cos_sim[i], i) for i in range(self.num_items)]
        cand.sort(lambda x,y: cmp(x[0],y[0]), reverse=True)
        #average new h from top-k h vectors, and predict with bpr
        i_factors = np.zeros(self.bpr_k)
        i_bias = 0
        for j in range(self.k):
            i_factors += cand[j][0] * self.bpr_model.item_factors[cand[j][1],:]
            i_bias += cand[j][0] * self.bpr_model.item_bias[cand[j][1]]
        sim_sum = sum(cand[j][0] for j in range(self.k))
        i_factors /= sim_sum
        i_bias /= sim_sum
        for u in range(self.num_users):
            result.append(i_bias + np.dot(self.bpr_model.user_factors[u], i_factors))
        return result

class Map_Linear(Mapper):

    def __init__(self, data, attr, bpr_k=None, bpr_args=None, learning_rate=None, penalty_factor=None):
        self.init(data, attr, bpr_k, bpr_args)
        self.learning_rate = learning_rate
        self.penalty_factor = penalty_factor

    def train(self, num_iters):
        self.bpr_model.train(self.data, self.sampler, num_iters)
        #train linear models for bpr_k column across attributes(X=attrs, Y=H[u])
        self.mapper_factors = np.random.random_sample((self.bpr_k, self.num_attrs))
        self.mapper_bias = np.zeros((self.bpr_k, 1))
        self.mapper_factors_b = np.random.random_sample(self.num_attrs)
        self.mapper_bias_b = np.zeros(1)
        for it in range(num_iters):
            print "Mapper Map_Linear trainning for iteration",it,"..."
            gradient = np.dot( \
                np.dot(self.mapper_factors, self.attr)+np.dot(self.mapper_bias, np.ones((1, self.num_items)))-self.bpr_model.item_factors.transpose() \
                , self.attr) \
                + self.penalty_factor*self.mapper_factors
            self.mapper_factors -= self.learning_rate/self.num_items*gradient
            gradient_b = np.dot( \
                np.dot(self.mapper_factors_b, self.attr)+self.mapper_bias_b*np.ones(self.num_items)-self.bpr_model.item_bias \
                , self.attr) \
                + self.penalty_factor*self.mapper_factors_b
            self.mapper_factors_b -= self.learning_rate/self.num_items*gradient_b

    def test(self, test_data, test_attr, prec_n=5):
        self.test_init(test_data, test_attr) 
        return [self.prec_at_n(prec_n), self.auc()]   

    def test_predict(self, i):
        result = []
        i_factors = self.mapper_bias + np.dot(self.mapper_factors, self.test_attr[i])
        i_bias = self.mapper_bias_b + np.dot(self.mapper_factors_b, self.test_attr[i])
        for u in range(self.num_users):
            result.append(i_bias + np.dot(self.bpr_model.user_factors[u], i_factors))
        return result

    def set_parameter(self, para_set):
        self.learning_rate = para_set[0]
        self.penalty_factor = para_set[1]

class Map_BPR(Mapper):

    def __init__(self, data, attr, bpr_k=None, bpr_args=None, learning_rate=None, penalty_factor=None):
        self.init(data, attr, bpr_k, bpr_args)
        self.learning_rate = learning_rate
        self.penalty_factor = penalty_factor

    def train(self, num_iters):
        self.bpr_model.train(self.data, self.sampler, num_iters)
        #train linear models for bpr_k column across attributes(X=attrs, Y=H[u])
        self.mapper_factors = np.random.random_sample((self.bpr_k, self.num_attrs))
        #self.mapper_bias = np.zeros((self.bpr_k, 1))
        #self.mapper_factors_b = np.random.random_sample(self.num_attrs)
        #self.mapper_bias_b = np.zeros(1)
        for it in range(num_iters):
            print "Mapper Map_BPR trainning for iteration",it,"..."
            for u,i,j in self.sampler.generate_samples(self.data):
                x_uij = self.predict(u,i) - self.predict(u,j)
                #XXX: maybe it should be exp(-x)/(1.0+exp(-x))
                #z = 1.0/(1.0+exp(x_uij))
                z = 1.0 - 1.0/(1.0+exp(-x_uij))
                u_factor = np.atleast_2d(self.bpr_model.user_factors[u,:]).transpose()
                ij_diff = np.atleast_2d(self.attr[i]-self.attr[j])

                gradient = z * np.dot(u_factor, ij_diff) 
                self.mapper_factors = self.learning_rate * ( \
                    gradient - self.penalty_factor * self.mapper_factors )
                #self.mapper_bias = self.learning_rate * ( \
                #    z * u_factor \
                #    - self.penalty_factor * self.mapper_bias )

    def predict(self, u, i):
        return np.dot( self.bpr_model.user_factors[u,:] \
            , np.dot(self.mapper_factors, self.attr[i]) )
            #\+self.mapper_bias )

    def test(self, test_data, test_attr, prec_n=5):
        self.test_init(test_data, test_attr) 
        return [self.prec_at_n(prec_n), self.auc()]   

    def test_predict(self, i):
        result = []
        i_factors = np.dot(self.mapper_factors, self.test_attr[i])
            #\+self.mapper_bias
        #no i_bias here because we didn't use actual h_i in trainning
        for u in range(self.num_users):
            result.append(np.dot(self.bpr_model.user_factors[u], i_factors))
        return result

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

    def test_predict(self, i):
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

    def test_predict(self, i, max_score=1.0):
        return [(random.random() * max_score) for i in range(self.num_users)]
