"""
Created on Sun Nov  8 16:49:11 2020

@author: ericrasmussen
"""


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

trainLabels15 = pd.read_csv("/Users/ericrasmussen/Desktop/ML and AI/Project/archive-1/labels/trainLabels15.csv").values
Test_labels15 = pd.read_csv("/Users/ericrasmussen/Desktop/ML and AI/Project/archive-1/labels/testLabels15.csv").values

Y_test = Test_labels15[:,1].tolist()
Tf = trainLabels15[:,1].tolist()


training_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory="/Users/ericrasmussen/Desktop/ML and AI/Project/archive-1/",
    labels= Tf,
    label_mode="int",
    validation_split=0.2,
    subset="training",
    seed = 123,
    color_mode="rgb",
    batch_size=32,
    image_size=(1024, 768)
)
val_data = tf.keras.preprocessing.image_dataset_from_directory(
  directory="/Users/ericrasmussen/Desktop/ML and AI/Project/archive-1/",
  labels= Tf,
  label_mode="int",
  validation_split=0.2,
  subset="validation",
  seed = 123,
  color_mode="rgb",
  batch_size=32,
  image_size=(1024, 768)
)
testing_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory="/Users/ericrasmussen/Desktop/ML and AI/Project/archive-2/",
    labels= Y_test,
    label_mode="int",
    color_mode="rgb",
    batch_size=32,
    image_size=(1024, 768)
)

print(training_data.list_files("/Users/ericrasmussen/Desktop/ML and AI/Project/archive-1/resized train 15/*.jpg"))
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
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(1024,768,3))
normalized_training_data = training_data.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_training_data))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image)) 

#normalizing RBG colors (1-255) for first batch
normalized_test_data = testing_data.map(lambda x, y: (normalization_layer(x), y))


#normalizing RBG colors (1-255) for first batch
normalized_val_data = val_data.map(lambda x, y: (normalization_layer(x), y))


AUTOTUNE = tf.data.experimental.AUTOTUNE
normalized_training_data = normalized_training_data.cache().prefetch(buffer_size=AUTOTUNE)
normalized_testing_data = normalized_test_data.cache().prefetch(buffer_size=AUTOTUNE)
normalized_val_data = normalized_val_data.cache().prefetch(buffer_size=AUTOTUNE)


model = tf.keras.Sequential([
  #layers.experimental.preprocessing.Rescaling(1./255, input_shape=(1024, 768, 3)),
  #layers.experimental.preprocessing.Resizing(
    #  1024, 768, interpolation='bilinear', name=None
#),
  layers.Conv2D(32, kernel_size =(5,5), activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, kernel_size =(3,3), activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(.25),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(.25),
  layers.Dense(num_classes,activation='softmax')
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])



model.fit(normalized_training_data,validation_data=normalized_val_data,epochs=3)
model.summary()
score = model.evaluate(X_test, Y_test, verbose=2)
