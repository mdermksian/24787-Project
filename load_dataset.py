import numpy as np
import pandas as pd
import tensorflow as tf

trainLabels15 = pd.read_csv("./archive/labels/trainLabels15.csv").values

print(type(trainLabels15[:,1].tolist()))


DATASET = tf.keras.preprocessing.image_dataset_from_directory(
    directory="archive/",
    labels=trainLabels15[:,1].tolist(),
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(1024, 768)
)

print(DATASET.list_files("archive/*.jpg"))