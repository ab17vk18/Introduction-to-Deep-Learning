# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Load the movies, users and ratings data
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python',
                     encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python',
                     encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python',
                     encoding='latin-1')

# Prepare training and test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Get the number of users and movies
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Convert the training and test data into an array with users as rows and 
# movies as columns
def convert(data):
    new_data = []
    for user_id in range(1,nb_users+1):
        movies_id = data[data[:,0]==user_id][:,1]
        ratings_by_user = data[data[:,0]==user_id][:,2]
        ratings = np.zeros(nb_movies)
        ratings[movies_id-1] = ratings_by_user
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Convert list of lists to torch.tensor
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Convert the ratings to binary values(1 if liked, 0 if not liked, -1 unwatched)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# RBM class - NN architecture
class RBM():
    
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
        
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
    
num_visible = len(training_set[0])
num_hidden = 128
batch_size = 64
rbm = RBM(num_visible, num_hidden)   
num_epoch = 20

# Train the RBM
for epoch in range(1, num_epoch + 1):
    train_loss = 0
    normalize_counter = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        # k steps contrastive divergence/gibbs sampling
        for k in range(16):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        # Calculate RMSE
        #train_loss += torch.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0]).pow(2)))
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        normalize_counter += 1.
    print("Epoch: {}, Loss: {}".format(
                                str(epoch), float(train_loss/normalize_counter)))
    #################Epoch: 20, Loss: 0.2457127869129181#######################
    
# Test the RBM
test_loss = 0
normalize_counter = 0.

for user_id in range(nb_users):
    v = training_set[user_id:user_id+1]
    vt = test_set[user_id:user_id+1]
    if len(vt[vt>=0]) > 0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        normalize_counter += 1.
        
print("Test Loss: {}".format(float(test_loss/normalize_counter)))
###########Test Loss: 0.2471938580274582########################