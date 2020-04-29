# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# Part 1 - Data Preprocessing

df = pd.read_csv("Churn_Modelling.csv")

X = df.iloc[:,3:13].values
y = df.iloc[:,13].values

# Encode text categorical variables
from sklearn.preprocessing import LabelEncoder
la_en = LabelEncoder()
X[:,1] = la_en.fit_transform(X[:,1])
X[:,2] = la_en.fit_transform(X[:,2])

X = X.astype(float)

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)

#Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Part 2 - Building the NN and train

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units=8, activation='relu', input_dim=10))
classifier.add(Dense(units=8, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Part 3 - Predictions and Evaluation

y_pred = classifier.predict(X_test)
y_pred_fin = y_pred > 0.5
y_pred_fin = y_pred_fin.astype(int)

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred_fin)

print("Accuracy on test set: {}".format(
        (conf_mat[0,0]+conf_mat[1,1])/len(y_pred_fin))
)

# Part 4 - Implementing K-Fold Cross Validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    
    classifier = Sequential()
    classifier.add(Dense(units=8, activation='relu', input_dim=10))
    classifier.add(Dense(units=8, activation='relu'))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy',
                       metrics=['accuracy'])
    return classifier
    
classifier = KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=-1)

mean_acc = np.mean(accuracies)
var_acc = (np.std(accuracies))**2

print(mean_acc, var_acc)

# Part 5 - Tuning hyperparameters using GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def grid_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=8, activation='relu', input_dim=10))
    classifier.add(Dense(units=8, activation='relu'))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy',
                       metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=grid_classifier,verbose=1)
parameters = {
        'batch_size' : [25,32],
        'epochs' : [200, 300],
        'optimizer' : ['Adam', 'rmsprop']        
        }
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, 
                           scoring='accuracy', cv=10)
grid_search = grid_search.fit(X_train,y_train)

best_params = grid_search.best_params_
best_acc = grid_search.best_score_

print(best_params)
print(best_acc)