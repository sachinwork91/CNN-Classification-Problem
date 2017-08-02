# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 20:38:34 2017

@author: Sachin
"""

#Main steps to Build CNN 
# Step 1 Convolution
# Step 2 Pooling 
# Step 3 Flattening
# Step 4 Fully Connected Layers


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#Initailise CNN
classifier=Sequential()

#Step 1: Convolution , First Argument of classifier is number of feature detectors of 3 by 3 
#dimensions which will result in 32 feature maps
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64 , 3),activation='relu'))

#Step 2: Pooling , this helps in reducing the size of the features 
classifier.add(MaxPooling2D(pool_size = (2,2)))


classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64 , 3),activation='relu'))

#Step 2: Pooling , this helps in reducing the size of the features 
classifier.add(MaxPooling2D(pool_size = (2,2)))



#Step 3 : Flattening
# if we perform flattening directly on the input image then we dont get 
# the spatial information of the pixels around a pixel 
#Also the number of pixels is high 
classifier.add(Flatten())

#Step 4 Adding Fully Connected Layer
#Hidden Layer
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dropout(0.5))
#Output Layer for Final Prediction
classifier.add(Dense(output_dim=64, activation='relu'))
classifier.add(Dropout(0.5))


classifier.add(Dense(output_dim=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#Performing Image Augmentation
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(  'dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary')

classifier.fit_generator(training_set,
                         samples_per_epoch=8000,
                         nb_epoch=25,
                         validation_data=test_set,
                         nb_val_samples=2000)










