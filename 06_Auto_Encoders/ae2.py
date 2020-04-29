# -*- coding: utf-8 -*-

'''
Predicting movie ratings by users using Stacked Autoencoders

'''

# Importing the libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the Stacked Autoencoder class
class SAE(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(nb_movies, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 32)
        self.fc4 = nn.Linear(32, nb_movies)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    
sae = SAE()
loss_fn = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.4)
       
# Training
num_epochs = 200
for epoch in range(1,num_epochs+1):
    train_loss = 0
    normalize_counter = 0.
    for user_id in range(nb_users):
        input = Variable(training_set[user_id]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data>0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = loss_fn(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data>0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data.item()*mean_corrector)
            normalize_counter += 1.
            optimizer.step()
    print("Epoch: {}, Loss: {}".format(epoch, train_loss/normalize_counter))
    ###############Epoch: 200, Loss: 0.886256430584516#########################
    
# Testing
test_loss = 0
normalize_counter = 0.
for user_id in range(nb_users):
    input = Variable(training_set[user_id]).unsqueeze(0)
    target = Variable(test_set[user_id]).unsqueeze(0)
    if torch.sum(target.data>0) > 0:
        output = sae(input)
        output[target == 0] = 0
        loss = loss_fn(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data>0) + 1e-10)
        test_loss += np.sqrt(loss.data.item()*mean_corrector)
        normalize_counter += 1.
print("Test loss: {}".format(test_loss/normalize_counter))
#########################Test loss: 0.9490992061551592#########################

# Save the model
curr_path = os.path.dirname('__file__')
model_path = os.path.join(curr_path, 'model', 'sae_movie_ratings_200ep.pkl')
torch.save(sae, model_path)
print("Model saved at the location", model_path)
    
            

        
    
    