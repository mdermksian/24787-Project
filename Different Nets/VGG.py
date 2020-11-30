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

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2500)])

photoDirectory = "./Compressed_Square_Structured/"
# photoDirectory = "./Cropped_Resized_Structured/"
# photoDirectory = "./Compressed_Square_Augment_Rotate_Structured/"

img_height = 330
img_width = 330

batchSize = 20


train_datagen = ImageDataGenerator(
  rescale = 1./255,
  featurewise_center = True,
  featurewise_std_normalization = True,
  rotation_range=20,
  width_shift_range = 0.2,
  brightness_range = [0.4, 1.5],
  height_shift_range = 0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  validation_split=0.15
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

# displays labels of batch 
# for image_batch, labels_batch in training_data:
  # print(image_batch.shape)
  # g = labels_batch.numpy()
  # break

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
# training_data = training_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#normalized_testing_data = normalized_test_data.cache().prefetch(buffer_size=AUTOTUNE)
# val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)


vgg16_model = keras.applications.vgg16.VGG16(input_shape = (img_height,img_width,3), weights = 'imagenet', include_top = False)
# model = Sequential()
# for layer in vgg16_model.layers[:-1]:
    # model.add(layer)
    
for layer in vgg16_model.layers:
    layer.trainable = False

x = Flatten()(vgg16_model.output)
x = Dense(num_classes, activation = 'softmax')(x)

# model.add(Dense(num_classes, activation='softmax'))

model = Model(inputs = vgg16_model.input, outputs = x)


# opt = SGD(lr=0.0001)
opt = 'adam'
model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer= tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

# model.summary()

runName = 'VGG16Net_Square_Resized_CUDA_20_Batch'
# runName = 'AlexNet_Square_Resized_CUDA_50_Batch'
epochLimit = 500

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs/'+runName)

# run tensorboard --logdir 'directory'
# model.fit_generator(training_data, steps_per_epoch = 1744, validation_data = val_data,validation_steps = 261, epochs = epochLimit, verbose = 2)

# model.fit(training_data,validation_data=val_data,epochs=epochLimit)
model.fit(training_data,validation_data=val_data,epochs=epochLimit, callbacks=[tensorboard_callback])

# AlexNet L2 Reg

# reg = regularizers.l1_l2()

# model = tf.keras.models.Sequential([
#     layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(img_height,img_width,3)),
#     layers.BatchNormalization(),
#     layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
#     layers.BatchNormalization(),
#     layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     layers.BatchNormalization(),
#     layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
#     layers.BatchNormalization(),
#     layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
#     layers.BatchNormalization(),
#     layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     layers.Flatten(),
#     layers.Dense(4096, activation='relu', kernel_regularizer=reg),
#     layers.Dropout(0.5),
#     layers.Dense(4096, activation='relu', kernel_regularizer=reg),
#     layers.Dropout(0.5),
#     layers.Dense(num_classes, activation='softmax', kernel_regularizer=reg)
# ])

# # lambda_ = 1
# # mainLoss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
# # regLoss = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
# # totalLoss = mainLoss + lambda_*regLoss

# opt = SGD(lr=0.001)
# model.compile(optimizer= opt, loss= tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
# # model.compile(optimizer= tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

# # model.summary()

# model.fit(training_data,validation_data=val_data,epochs=500)