# This script takes the large training set, finds faces, crops to those faces, and resizes the image. 
# After many attempts, if no face is found, the image is omitted from the training set.

import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image

# Define folder paths
pics_path = r'C:\Users\shawn\Documents\Kaggle\train' #\test            # large folder with all (color) training data
kaggle_path = r'C:\Users\shawn\Documents\Kaggle'                 # top level folder
processed_path = r'C:\Users\shawn\Documents\Kaggle\train_color' #\test_color# folder to put processed training data in
validation_path = r'C:\Users\shawn\Documents\Kaggle\validation_color'  # folder to put processed validation data in
    
# import face recognition tool
os.chdir(kaggle_path)
face_cascade = cv2.CascadeClassifier('face_rec.xml')

os.chdir(kaggle_path)
allCategories = pd.read_csv('category.csv')
trainData = pd.read_csv('train.csv')
categories = allCategories['Category'].tolist()
allFileNames = (trainData['File Name']).tolist()

def getCelebName(pic): # returns the celebrity's name (and ID#) from 'XXXXX.png'
    celebName = trainData['Category'][allFileNames.index(pic)]
    celebID = categories.index(celebName)
    return [celebName, celebID]

def cropToFace(pic): # Returns the cropped image if any faces are detected
    os.chdir(pics_path)
    im = Image.open(pic).convert('RGB')
    img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    layers = 3 
    scale = 1.2 
    cropSize = 64
    width = np.min(img.shape)
    minWidth = width
    faces = face_cascade.detectMultiScale(img, scale, layers, minSize = (minWidth, minWidth))
    
    # try REALLY hard to find a face O-><-O
    while len(faces) < 1 and layers > 1 and scale > 1.1 and minWidth > width//5 :
        minWidth -= 10
        faces = face_cascade.detectMultiScale(img, scale, layers, minSize = (minWidth, minWidth))
    width = np.min(img.shape)
    minWidth = width//20
    scale = 1.5
    layers = 8
    while len(faces) < 1 and layers > 2:
        layers -= 1
        faces = face_cascade.detectMultiScale(img, scale, layers, minSize = (minWidth, minWidth))

    if len(faces) > 0:
        x, y, w, h = faces[0,:]
        im = im.crop((x, y, x+w, y+h))
    cropSize = 160
    im = im.resize((cropSize, cropSize))
    return [len(faces)>0, im]

pics = sorted(os.listdir(pics_path),key=lambda x: int(os.path.splitext(x)[0]))


i = 0 # number of pictures processed for training
j = 0 # number of pictures saved for validation
k = 0 # number of pictures skipped (error loading or no face found)
validationFraction = 0.1 # ratio of training data saved for validation

numPics = len(pics)

for pic in pics:
    if ((i+j+k) % 1000) == 0:
        print(str(100*round((i+j+k)/numPics,2)) + '%, training = ' + str(i) + ', validation = ' + str(j) + ', skipped = ' + str(k))    
    [faceFound, img] = cropToFace(pic)
    if (faceFound):
        if (i+j+k)/numPics < (1 - validationFraction):
            os.chdir(processed_path)
            i += 1
        else:
            os.chdir(validation_path)
            j += 1
        [celebName, celebID] = getCelebName(pic)
        if not os.path.isdir(celebName):
            os.mkdir(celebName)
        os.chdir(celebName)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(pic, img) # save the image into the processed folder
    else: 
        k += 1
print('training = ' + str(i) + ', validation = ' + str(j) + ', skipped = ' + str(k))    
    
