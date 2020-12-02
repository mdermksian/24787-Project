import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image
import os, sys
from keras.optimizers import SGD
from sklearn.utils import compute_class_weight
import tensorflow.compat.v1.keras.backend as K
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from tensorflow.keras import regularizers
from keras.layers.convolutional import *
from keras.metrics import categorical_crossentropy
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
import keras
from keras.models import Model
from collections import Counter

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2500)])

photoDirectory = "./Gaussian_Filtered_Class/"
# photoDirectory = "./Equal_Class_Split_Filtered_Color/"
# photoDirectory = "./Compressed_Square_Structured/"
# photoDirectory = "./Cropped_Resized_Structured/"
# photoDirectory = "./Compressed_Square_Augment_Rotate_Structured/"

img_height = 224
img_width = 224

batchSize = 20


train_datagen = ImageDataGenerator(
  rescale = 1./255,
  featurewise_center = False,
  featurewise_std_normalization = False,
  rotation_range=10,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  validation_split=0.20
)

training_data = train_datagen.flow_from_directory(
    photoDirectory,
    target_size=(img_height, img_width),
    batch_size=batchSize,
    class_mode='categorical',
    color_mode="rgb",
    subset='training'
)
val_data = train_datagen.flow_from_directory(
  photoDirectory,
    target_size=(img_height, img_width),
    batch_size=batchSize,
    class_mode='categorical',
    color_mode="rgb",
    subset='validation'
)

# print(training_data.list_files("/Users/ericrasmussen/Desktop/ML and AI/Project/archive-1/resized train 15/*.jpg"))
num_classes = 5



#buffering to avoid bottlenecks

# AUTOTUNE = tf.data.experimental.AUTOTUNE
# training_data = training_data.cache()
# testing_data = testing_data.cache()
# val_data = val_data.cache()
# training_data = training_data.prefetch(buffer_size=AUTOTUNE)
# val_data = training_data.prefetch(buffer_size=AUTOTUNE)
# testing_data = testing_data.prefetch(buffer_size=AUTOTUNE)
# num_classes = 5


AUTOTUNE = tf.data.experimental.AUTOTUNE


vgg16_model = keras.applications.vgg16.VGG16(input_shape = (img_height,img_width,3), weights = 'imagenet') #include_top = True
# model = Sequential()
# for layer in vgg16_model.layers[:-1]:
    # model.add(layer)
vgg16_model.summary()
x = vgg16_model.get_layer('fc2').output
prediction = Dense(5, activation = 'softmax', name = 'predictions')(x)

model = Model(inputs = vgg16_model.input, outputs = prediction)

for layer in model.layers:
    layer.trainable = False

for layer in model.layers[-16:]:
    layer.trainable = True
    
# x = Flatten()(vgg16_model.output)
# x = Dense(num_classes, activation = 'softmax')(x)

# model.add(Dense(num_classes, activation='softmax'))

# model = Model(inputs = vgg16_model.input, outputs = x)


# opt = SGD(lr=0.0001)
opt =  Adam(lr = 0.000001)
model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer= tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

# model.summary()

#Class Weights
counter = Counter(training_data.classes)
max_val = float(max(counter.values()))
class_weights = {class_id: max_val/num_images for class_id, num_images in counter.items()}


runName = 'VGG16Net_Gaussian_LR_0.000001'
# runName = 'AlexNet_Square_Resized_CUDA_50_Batch'
epochLimit = 500

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs/'+runName)

# run tensorboard --logdir 'directory'
# model.fit_generator(training_data, steps_per_epoch = 1744, validation_data = val_data,validation_steps = 261, epochs = epochLimit, verbose = 2)

# model.fit(training_data,validation_data=val_data,epochs=epochLimit)
# model.fit(training_data,validation_data=val_data,epochs=epochLimit, callbacks=[tensorboard_callback])

hist = model.fit(training_data, steps_per_epoch = training_data.samples//training_data.batch_size,validation_data = val_data, validation_steps = val_data.samples//val_data.batch_size, epochs = epochLimit, callbacks = [tensorboard_callback])
