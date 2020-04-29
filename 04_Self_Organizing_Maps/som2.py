# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# Import the dataset
df = pd.read_csv("Credit_Card_Applications.csv")
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som_model = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som_model.random_weights_init(X)
som_model.train_random(data=X, num_iteration=200)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som_model.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i,x in enumerate(X):
    w = som_model.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)   
show()

# Cherry-picking the frauds
mappings = som_model.win_map(X)
likely_frauds = mappings[(8,5)]
likely_frauds = scaler.inverse_transform(likely_frauds)
