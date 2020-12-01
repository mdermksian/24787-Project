#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 21:28:01 2020

@author: ericrasmussen
"""


from PIL import Image
import os, sys

OutputLocation = "D:/GitHub/Kaggle_Unstructured/"

StockPhotoLocation = "./archive/resized train 15/"

# path = "/Users/ericrasmussen/Desktop/ML and AI/Project/Messydor/Original_data_not_comp/"
dirs = os.listdir( StockPhotoLocation )

def resize(OutputLocation):
    for item in dirs:
        if item == '.DS_Store':
            continue
        else:
            if os.path.isfile(StockPhotoLocation+item):
                im = Image.open(StockPhotoLocation+item)
                f, e = os.path.splitext(StockPhotoLocation+item)
                imResize = im.resize((330,330), Image.ANTIALIAS)
                # imResize = imResize.crop((90,10,420,340))
                newFileName = f.replace("./archive/resized train 15/","")

                print(newFileName)

                imResize.save(OutputLocation+newFileName + '.jpg', 'JPEG', quality=90)
        


resize(OutputLocation)

# def crop():
#     for item in dirs:
#         if item == '.DS_Store':
#             continue
#         else:
#             if os.path.isfile(path+item):
#                 im = Image.open(path+item)
#                 f, e = os.path.splitext(path+item)
#                 imCrop = im.crop((90,10,90,5))
#                 imCrop.save(f + '_crop.jpg', 'JPEG', quality=100)

# crop()
