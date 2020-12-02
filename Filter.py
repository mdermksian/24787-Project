#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:19:53 2020

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
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# img = cv.imread('57_left.jpg',0)
# hist,bins = np.histogram(img.flatten(),256,[0,256])
# cdf = hist.cumsum()
# cdf_normalized = cdf * float(hist.max()) / cdf.max()
# plt.plot(cdf_normalized, color = 'b')
# plt.hist(img.flatten(),256,[0,256], color = 'r')
# plt.xlim([0,256])
# plt.legend(('cdf','histogram'), loc = 'upper left')
# plt.show()
# plt.figure()
# plt.plot(img)
# cdf_m = np.ma.masked_equal(cdf,0)
# cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
# cdf = np.ma.filled(cdf_m,0).astype('uint8')
# plt.figure()
# img2 = cdf[img]


# hist,bins = np.histogram(img2.flatten(),256,[0,256])
# plt.hist(img2.flatten(),256,[0,256], color = 'r')
# plt.plot(cdf, color = 'b')
# plt.xlim([0,256])

# img = cv.imread('57_left.jpg',0)
# equ = cv.equalizeHist(img)
# res = np.hstack((img,equ)) #stacking images side-by-side
# cv.imwrite('res.png',res)


# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe.apply(img)
# cv.imwrite('clahe_2.jpg',cl1)

def Filter_images_Black_White(directory='./4'):
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
        img = cv.imread(image,0)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(img)
        cv.imwrite("Filtered"+image,cl1)

def Filter_images_Color(directory='./4'):
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
        bgr = cv.imread(image)
        lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
        lab_planes = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv.merge(lab_planes)
        bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
        cv.imwrite("Filtered_Color"+image,bgr)
        

        
Filter_images_Color()

#Filter_images_Black_White()