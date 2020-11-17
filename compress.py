#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 08:40:07 2020

@author: ericrasmussen
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import os
import glob

def compress_images(directory='/Users/ericrasmussen/Desktop/ML and AI/Project/Messydor/Messidor-small/IMAGES-1', quality=5):
    # 1. If there is a directory then change into it, else perform the next operations inside of the 
    # current working directory:
    if directory:
        os.chdir(directory)

    # 2. Extract all of the .png and .jpeg files:
    files = os.listdir()

    # 3. Extract all of the images:
    images = [file for file in files if file.endswith(('jpg', 'png'))]

    # 4. Loop over every image:
    for image in images:
        print(image)

        # 5. Open every image:
        img = Image.open(image)

        # 5. Compress every image and save it with a new name:
        img.save("Compressed_"+image, optimize=True, quality=quality)
        
        
compress_images()