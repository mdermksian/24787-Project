#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:49:11 2020

@author: ericrasmussen
"""


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image
import os, sys
from keras.optimizers import SGD
from sklearn.utils import compute_class_weight


trainLabels15 = pd.read_csv("/Users/ericrasmussen/Desktop/ML and AI/Project/Messydor/Messidor-small/messy_data_small.csv").values
#Test_labels15 = pd.read_csv("/Users/ericrasmussen/Desktop/ML and AI/Project/archive-1/labels/testLabels15.csv").values

#Y_test = Test_labels15[:,1].tolist()
lab = trainLabels15[:,1].tolist()
Tf = lab

training_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory="/Users/ericrasmussen/Desktop/ML and AI/Project/Messydor/Messidor-small/",
    labels= Tf,
    label_mode="int",
    validation_split=0.2,
    subset="training",
    seed = 123,
    color_mode="rgb",
    batch_size=16,
    image_size=(330, 335)
)
val_data = tf.keras.preprocessing.image_dataset_from_directory(
  directory="/Users/ericrasmussen/Desktop/ML and AI/Project/Messydor/Messidor-small/",
  labels= Tf,
  label_mode="int",
  validation_split=0.2,
  subset="validation",
  seed = 123,
  color_mode="rgb",
  batch_size=16,
  image_size=(330, 335)
)
# testing_data = tf.keras.preprocessing.image_dataset_from_directory(
#     directory="/Users/ericrasmussen/Desktop/ML and AI/Project/archive-2/",
#     labels= Y_test,
#     label_mode="int",
#     color_mode="rgb",
#     batch_size=10,
#     image_size=(512, 512)
# )

# print(training_data.list_files("/Users/ericrasmussen/Desktop/ML and AI/Project/archive-1/resized train 15/*.jpg"))
num_classes = 5

# object_methods = [method_name for method_name in dir(training_data)
#                   if callable(getattr(training_data, method_name))]

#Plotting figures with associated label
# plt.figure(figsize=(10, 10))
# for images, labels in training_data.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(Tf[i])
#     plt.axis("off")

#displays labels of batch 
for image_batch, labels_batch in training_data:
  print(image_batch.shape)
  g = labels_batch.numpy()
  break

#buffering to avoid bottlenecks

# AUTOTUNE = tf.data.experimental.AUTOTUNE
# training_data = training_data.cache()
# testing_data = testing_data.cache()
# val_data = val_data.cache()
# training_data = training_data.prefetch(buffer_size=AUTOTUNE)
# val_data = training_data.prefetch(buffer_size=AUTOTUNE)
# testing_data = testing_data.prefetch(buffer_size=AUTOTUNE)
# num_classes = 5

#normalizing RBG colors (1-255) for first batch
# normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(510,340,3))
# normalized_training_data = training_data.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_training_data))
# first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image)) 

#normalizing RBG colors (1-255) for first batch
#normalized_test_data = testing_data.map(lambda x, y: (normalization_layer(x), y))


#normalizing RBG colors (1-255) for first batch
#normalized_val_data = val_data.map(lambda x, y: (normalization_layer(x), y))


AUTOTUNE = tf.data.experimental.AUTOTUNE
training_data = training_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#normalized_testing_data = normalized_test_data.cache().prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)


model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(330, 335, 3)),
  layers.Conv2D(80, kernel_size =(11,11),strides=(3,3), input_shape=(330,335,3), activation='relu', padding= 'same'),
  layers.BatchNormalization(),
  layers.MaxPooling2D((3,3),strides=(2,2)),
  layers.Conv2D(160, kernel_size =(5,5), activation='relu',padding= 'same'),
  layers.BatchNormalization(),
  layers.AveragePooling2D((3,3),strides=(2,2)),
  layers.Conv2D(320, (3,3), activation='relu',padding= 'same'),
  layers.BatchNormalization(),
  layers.MaxPooling2D((3,3)),
  layers.Conv2D(160, kernel_size =(5,5), activation='relu',padding= 'same'),
  layers.BatchNormalization(),
  layers.MaxPooling2D((3,3),strides=(2,2)),
  layers.Flatten(),
  layers.Dense(5000, activation='relu'),
  layers.Dropout(.50),
  layers.Dense(2500, activation=('relu')),
  layers.Dropout(.50),
  layers.Dense(500, activation='sigmoid'),
  layers.Dropout(.25),
  layers.Dense(num_classes,activation='relu')
])
opt = SGD(lr=0.001)
model.compile(
  optimizer= opt,
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.summary()
outputLabels = np.array([[0, 1, 2, 3, 4]]).reshape(-1)
outputs = labels_batch.numpy()
classWeight = compute_class_weight('balanced', outputLabels, Tf) 
classWeight = dict(enumerate(classWeight))
model.fit(training_data,validation_data=val_data,epochs=100,class_weight=classWeight)
#score = model.evaluate(X_test, Y_test, verbose=1)



