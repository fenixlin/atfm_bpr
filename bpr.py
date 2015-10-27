"""
Copyright (c) 2015 Fenix Lin

Bayesian Personalized Ranking

Matrix Factorization model and a variety of classes
implementing different sampling strategies.
"""

import numpy as np
from math import exp, log
from copy import deepcopy
import random

class BPRArgs(object):

    def __init__(self,learning_rate=0.05,
                 bias_regularization=1.0,
                 user_regularization=0.0025,
                 positive_item_regularization=0.0025,
                 negative_item_regularization=0.00025,
                 update_negative_item_factors=True):
        self.learning_rate = learning_rate
        self.bias_regularization = bias_regularization
        self.user_regularization = user_regularization
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization
        self.update_negative_item_factors = update_negative_item_factors

class BPR(object):

    def __init__(self,D,args):
        """initialise BPR matrix factorization model
        D: number of factors
        """
        self.D = D
        self.learning_rate = args.learning_rate
        self.bias_regularization = args.bias_regularization
        self.user_regularization = args.user_regularization
        self.positive_item_regularization = args.positive_item_regularization
        self.negative_item_regularization = args.negative_item_regularization
        self.update_negative_item_factors = args.update_negative_item_factors

    def train(self,dataidx,num_items,sampler,num_iters):
        """train model
        data: user-item matrix as a scipy sparse matrix
              users and items are zero-indexed
        """
        self.init(dataidx,num_items)

        #print 'initial loss = {0}'.format(self.loss())
        for it in xrange(num_iters):
            #print 'starting iteration {0}'.format(it)
            for u,i,j in sampler.generate_samples(self.dataidx, self.num_items):
                self.update_factors(u,i,j)
            print 'iteration {0}: loss = {1}'.format(it,self.loss())

    def init(self,dataidx,num_items):
        self.dataidx = dataidx
        self.num_users = len(dataidx)
        self.num_items = num_items

        self.item_bias = np.zeros(self.num_items)
        self.user_factors = np.random.random_sample((self.num_users,self.D))
        self.item_factors = np.random.random_sample((self.num_items,self.D))

        self.create_loss_samples()

    def create_loss_samples(self):
        # apply rule of thumb to decide num samples over which to compute loss
        num_loss_samples = int(100*self.num_users**0.5)

        sampler = UniformUserUniformItem()
        self.loss_samples = [t for t in sampler.generate_samples(self.dataidx,self.num_items,num_loss_samples)]

    def update_factors(self,u,i,j,update_u=True,update_i=True):
        """apply SGD update"""
        update_j = self.update_negative_item_factors

        x = self.item_bias[i] - self.item_bias[j] \
            + np.dot(self.user_factors[u,:],self.item_factors[i,:]-self.item_factors[j,:])

        #XXX: maybe it should be exp(-x)/(1.0+exp(-x))
        #z = 1.0/(1.0+exp(x))
        z = 1.0 - 1.0/(1.0+exp(-x))

        # update bias terms
        if update_i:
            d = z - self.bias_regularization * self.item_bias[i]
            self.item_bias[i] += self.learning_rate * d
        if update_j:
            d = -z - self.bias_regularization * self.item_bias[j]
            self.item_bias[j] += self.learning_rate * d

        if update_u:
            d = (self.item_factors[i,:]-self.item_factors[j,:])*z - self.user_regularization*self.user_factors[u,:]
            self.user_factors[u,:] += self.learning_rate*d
        if update_i:
            d = self.user_factors[u,:]*z - self.positive_item_regularization*self.item_factors[i,:]
            self.item_factors[i,:] += self.learning_rate*d
        if update_j:
            d = -self.user_factors[u,:]*z - self.negative_item_regularization*self.item_factors[j,:]
            self.item_factors[j,:] += self.learning_rate*d

    def loss(self):
        ranking_loss = 0;
        for u,i,j in self.loss_samples:
            x = self.predict(u,i) - self.predict(u,j)
            #it should be ln(1.0/(1.0+exp(-x)) according to thesis)
            #ranking_loss += 1.0/(1.0+exp(x))
            ranking_loss += log(1.0/(1.0+exp(-x)))

        complexity = 0;
        for u,i,j in self.loss_samples:
            complexity += self.user_regularization * np.dot(self.user_factors[u],self.user_factors[u])
            complexity += self.positive_item_regularization * np.dot(self.item_factors[i],self.item_factors[i])
            complexity += self.negative_item_regularization * np.dot(self.item_factors[j],self.item_factors[j])
            complexity += self.bias_regularization * self.item_bias[i]**2
            complexity += self.bias_regularization * self.item_bias[j]**2

        #XXX: where does 0.5 come from? returns negative BPR-OPT so that it looks we are minimizing it
        #return ranking_loss + 0.5*complexity
        return -ranking_loss + complexity

    def predict(self,u,i):
        return self.item_bias[i] + np.dot(self.user_factors[u],self.item_factors[i])

# sampling strategies

class Sampler(object):

    def __init__(self):
        pass

    def init(self,dataidx,num_items,max_samples=None):
        self.dataidx = dataidx
        self.num_users = len(dataidx)
        self.num_items = num_items
        self.max_samples = max_samples
        self.datannz = 0
        for u in range(self.num_users):
            self.datannz += len(dataidx[u])

    def sample_user(self):
        u = self.uniform_user()
        num_pos = len(self.dataidx[u])
        assert(num_pos > 0 and num_pos != self.num_items)
        return u

    def sample_negative_item(self,user_items):
        j = random.randint(0,self.num_items-1)
        while j in user_items:
            j = random.randint(0,self.num_items-1)
        return j

    def uniform_user(self):
        return random.randint(0,self.num_users-1)

    def num_samples(self,n):
        if self.max_samples is None:
            return n
        return min(n,self.max_samples)

class UniformUserUniformItem(Sampler):

    def generate_samples(self,dataidx,num_items,max_samples=None):
        self.init(dataidx,num_items,max_samples)
        for _ in xrange(self.num_samples(self.datannz)):
            u = self.uniform_user()
            indices = self.dataidx[u]
            # sample positive item
            num_pos = len(indices)
            if (num_pos<=0 or num_pos==self.num_items):
                #throw bad user samples out
                continue
            i = random.choice(indices)
            j = self.sample_negative_item(indices)
            yield u,i,j

class UniformUserUniformItemWithoutReplacement(Sampler):

    def generate_samples(self,dataidx,num_items,max_samples=None):
        self.init(dataidx,num_items,max_samples)
        # make a local copy of data as we're going to "forget" some entries
        self.local_dataidx = deepcopy(self.dataidx)
        for _ in xrange(self.num_samples(self.datannz)):
            u = self.uniform_user()
            # sample positive item without replacement if we can
            user_items = self.local_dataidx[u]
            if user_items.size == 0:
                if self.dataidx[u].size == 0:
                    continue
                # reset user data if it's all been sampled
                self.local_dataidx[u] = self.dataidx[u].copy()
                user_items = self.local_dataidx[u]
            i = random.randint(0,user_items.size-1)
            # forget this item so we don't sample it again for the same user
            self.local_dataidx[u] = np.delete(user_items,i)
            j = self.sample_negative_item(user_items)
            yield u,i,j

class ExternalSchedule(Sampler):

    def __init__(self,filepath,index_offset=0):
        self.filepath = filepath
        self.index_offset = index_offset

    def generate_samples(self,dataidx,num_items,max_samples=None):
        self.init(dataidx,num_items,max_samples)
        f = open(self.filepath)
        samples = [map(int,line.strip().split()) for line in f]
        random.shuffle(samples)  # important!
        num_samples = self.num_samples(len(samples))
        for u,i,j in samples[:num_samples]:
            yield u-self.index_offset,i-self.index_offset,j-self.index_offset
