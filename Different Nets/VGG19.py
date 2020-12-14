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
import seaborn as sns
import io

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2500)])

photoDirectory = "./Gaussian_Filtered_Class/"
# photoDirectory = "./Equal_Class_Split_Filtered_Color/"
# photoDirectory = "./Compressed_Square_Structured/"
# photoDirectory = "./Cropped_Resized_Structured/"
# photoDirectory = "./Compressed_Square_Augment_Rotate_Structured/"
# photoDirectory = "./Gaussian_Color_Filters_Equal/"


img_height = 224
img_width = 224

batchSize = 32


train_datagen = ImageDataGenerator(
  rescale = 1./255,
  # featurewise_center = False,
  # featurewise_std_normalization = False,
  # rotation_range=10,
  # width_shift_range = 0.2,
  # height_shift_range = 0.2,
  # zoom_range=0.2,
  # horizontal_flip=True,
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

# VGG 19 Model
class VGG19(Sequential):
    def __init__(self, input_shape):
        super().__init__()

        self.add(Conv2D(64, kernel_size=(3,3), padding= 'same',
                        activation= 'relu', input_shape= input_shape))
        self.add(Conv2D(64, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))

        self.add(Conv2D(128, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
        self.add(Conv2D(128, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))

        self.add(Conv2D(256, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
        self.add(Conv2D(256, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
        self.add(Conv2D(256, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
        self.add(Conv2D(256, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))

        self.add(Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
        self.add(Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
        self.add(Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
        self.add(Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))

        self.add(Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
        self.add(Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
        self.add(Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
        self.add(Conv2D(512, kernel_size=(3,3), padding= 'same',
                        activation= 'relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))

        self.add(Flatten())
        self.add(Dense(4096, activation= 'relu'))
        self.add(Dropout(0.5))
        self.add(Dense(4096, activation= 'relu'))
        self.add(Dropout(0.5))
        self.add(Dense(num_classes, activation= 'softmax'))

        
                    
                    
model = VGG19((224, 224, 3))


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

num_classes = 5
classes=[0,1,2,3,4]

    
# x = Flatten()(VGG19_model.output)
# x = Dense(num_classes, activation = 'softmax')(x)

# model = Model(inputs = VGG19_model.input, outputs = x)


# opt = SGD(lr=0.0001)
opt =  Adam(lr = 0.0003)
model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer= tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()

#Class Weights
counter = Counter(training_data.classes)
max_val = float(max(counter.values()))
class_weights = {class_id: max_val/num_images for class_id, num_images in counter.items()}


runName = 'VGG19_Gaussian_Confusion_Matrix_1'
# runName = 'AlexNet_Square_Resized_CUDA_50_Batch'
epochLimit = 500


file_writer = tf.summary.create_file_writer('./logs/' + runName + '/cm')

def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
  test_pred = model.predict_classes(val_data)

  con_mat = tf.math.confusion_matrix(labels=val_data.labels, predictions=test_pred).numpy()
  con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

  con_mat_df = pd.DataFrame(con_mat_norm,
                     index = classes, 
                     columns = classes)

  figure = plt.figure(figsize=(8, 8))
  sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  
  buf = io.BytesIO()
  plt.savefig(buf, format='png')

  plt.close(figure)
  buf.seek(0)
  image = tf.image.decode_png(buf.getvalue(), channels=4)

  image = tf.expand_dims(image, 0)
  
  # Log the confusion matrix as an image summary.
  with file_writer.as_default():
    tf.summary.image("Confusion Matrix", image, step=epoch)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs/'+runName)

cm_callback =tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

# run tensorboard --logdir 'directory'
# model.fit_generator(training_data, steps_per_epoch = 1744, validation_data = val_data,validation_steps = 261, epochs = epochLimit, verbose = 2)

# model.fit(training_data,validation_data=val_data,epochs=epochLimit)
# model.fit(training_data,validation_data=val_data,epochs=epochLimit, callbacks=[tensorboard_callback])

hist = model.fit(training_data, steps_per_epoch = training_data.samples//training_data.batch_size,validation_data = val_data, 
validation_steps = val_data.samples//val_data.batch_size,
epochs = epochLimit,class_weight = class_weights, callbacks = [tensorboard_callback, cm_callback])
