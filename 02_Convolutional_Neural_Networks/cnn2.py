# -*- coding: utf-8 -*-

# Step 1: Building the CNN
from keras.models import Sequential
from keras.layers import (Conv2D,
                          MaxPooling2D,
                          Flatten,
                          Dense,
                          Dropout)

classifier = Sequential()

# Conv2D 1
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), 
                      input_shape=(32, 32, 3), activation='relu'))

# Pooling 1
classifier.add(MaxPooling2D(pool_size=(2,2), strides=None))

# Conv2D 2
classifier.add(Conv2D(filters=32, kernel_size=(3,3),
                      activation='relu'))

# Pooling 2
classifier.add(MaxPooling2D(pool_size=(2,2), strides=None))

# Flatten
classifier.add(Flatten())

# Fully Connected
classifier.add(Dense(units=64, activation='relu'))

# Dropout to reduce overfitting
classifier.add(Dropout(0.3))

# Output
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile
classifier.compile(optimizer='adam', loss='binary_crossentropy', 
                   metrics=['accuracy'])

# Step 2: Keras Image Preprocessing and Fitting the model
#https://keras.io/preprocessing/image/ - .flow_from_directory(directory)
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(32, 32),
        batch_size=16,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(32, 32),
        batch_size=16,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=5000,
        epochs=10,
        validation_data=test_set,
        validation_steps=1000)

# Step 3: Saving the model
import os
curr_dir = os.path.dirname('__file__')
save_model_path = os.path.join(curr_dir,'model/cat_or_dog_10ep.h5')
classifier.save(save_model_path)
print("Model saved as {}".format(save_model_path))

# Step 4: Making a single prediction:
import numpy as np
from keras.preprocessing import image

test_img = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',
                          target_size=(32,32))
test_img = image.img_to_array(test_img) #(32,32) to (32,32,3)
test_img = np.expand_dims(test_img, axis=0) #(32,32,3) to (1,32,32,3)
result = classifier.predict(test_img)

#print(training_set.class_indices)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print("It is a {}".format(prediction))