#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 21:28:01 2020

@author: ericrasmussen
"""


from PIL import Image
import os, sys

path = "/Users/ericrasmussen/Desktop/ML and AI/Project/Messydor/Original_data_not_comp/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if item == '.DS_Store':
            continue
        else:
            if os.path.isfile(path+item):
                im = Image.open(path+item)
                f, e = os.path.splitext(path+item)
                imResize = im.resize((510,340), Image.ANTIALIAS)
                imResize = imResize.crop((90,10,420,345))
                imResize.save(f + '.jpg', 'JPEG', quality=90)
        


resize()

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
