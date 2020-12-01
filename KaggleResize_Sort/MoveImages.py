import shutil
import os
import pandas as pd
import numpy as np

# data = pd.read_csv('./archive2/messidor_data.csv')
# data = data.to_numpy()

# data = data[~np.isnan(data[:,1].tolist()), :]

data = pd.read_csv('./archive/labels/trainLabels15.csv')

data = data.to_numpy()

OutputLocation = "./Kaggle/"

# StockPhotoLocation = "./Images-Croped-resized/"

# StockPhotoLocation = "D:/GitHub/archive/resized train 15/"

StockPhotoLocation = "D:/GitHub/Kaggle_Unstructured/"

for i in range(data.shape[0]):
    string_int_label = str(int(data[i,1]))

    fileName = data[i,0]

    fileName = fileName + ".jpg"

    # print(fileName,string_int_label)

    for i in range(1):
        # newFileName = fileName.replace(".png","")
        # newFileName = newFileName.replace(".jpg","")


        # newFileName = newFileName + '_' + str(i*90) + '.jpg'

        newFileName = fileName.replace(".png",".jpg")

        print(newFileName,string_int_label)

        os.makedirs(OutputLocation+string_int_label, exist_ok=True)
        shutil.copy2(StockPhotoLocation+newFileName, OutputLocation+string_int_label)