# This script takes the large training set and splits it into folders by celebrity. It reserves the last x% for validation

import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt

# Define folder paths
pics_path = r'C:\Users\shawn\Documents\Kaggle\train_small'
kaggle_path = r'C:\Users\shawn\Documents\Kaggle'
processed_path = r'C:\Users\shawn\Documents\Kaggle\train_sorted'
validation_path = r'C:\Users\shawn\Documents\Kaggle\validation'

# import face recognition tool
os.chdir(kaggle_path)
face_cascade = cv2.CascadeClassifier('face_rec.xml')

# function definitions
def getFaces(pic):
    os.chdir(pics_path)
    img = cv2.imread(pic)                                         # load the picture into an array
    if img is not None:                                           # only continue if the picture was succesfully loaded
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)              # convert the image to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.2, 8)       # attempt to detect any face(s) (set to be very strict!)
        faceFound = len(faces) > 0
        if faceFound:                                             # if any faces were detected, ...
            x, y, w, h = faces[0,:]                               # extract the position and size of the face
            cropped = gray[y:y+h, x:x+w]                          # store a cropped version of the image using the face rectangle
            cropped_size = 64                                     # set the size of all images
            processed = cv2.resize(cropped, (cropped_size, cropped_size)) # resize all images to be the same dimensions
        else: processed = -1
    else: 
        faceFound = False
        processed = -1
    return [faceFound, processed]                                 # faceFound = boolean, processed = final image

def getCelebID(pic):                                              # returns the celebrity's name (and ID#) from 'XXXXX.png'
    celebName = trainData['Category'][allFileNames.index(pic)]
    celebID = categories.index(celebName)
    return [celebName, celebID]
    
# Process data
# 1. Load pictures from the big data file 
# 2. Process - make grayscale, find faces, crop
# 3. Sort into folders based on the celebrity
# 4. Reserve the last 10-20% (specified) for validation

#load list of all pics
os.chdir(pics_path)
pics = os.listdir(pics_path)
numPics = len(pics)

# load table which has each pic name and matching celeb name
os.chdir(kaggle_path)
allCategories = pd.read_csv('category.csv')
trainData = pd.read_csv('train_small.csv')
categories = allCategories['Category'].tolist()
allFileNames = (trainData['File Name']).tolist()

i = 0 # number of pictures processed for training
j = 0 # number of pictures saved for validation
k = 0 # number of pictures skipped (error loading or no face found)
validationFraction = 0.2 # ratio of training data saved for validation

numPics = len(pics)

for pic in pics:
    if ((i+j+k) % 100) == 0:
        print(round((i+j+k)/numPics,2))
    [faceFound, processed] = getFaces(pic)
    if (faceFound):
        if (i+j+k)/numPics < (1 - validationFraction):
            os.chdir(processed_path)
            i += 1
        else:
            os.chdir(validation_path)
            j += 1
        [celebName, celebID] = getCelebID(pic)
        if not os.path.isdir(celebName):
            os.mkdir(celebName)
        os.chdir(celebName)
        cv2.imwrite(pic, processed) # save the image into the processed folder
        
    else: 
        k += 1
print('testing = ' + str(i) + ', validation = ' + str(k) + ', skipped = ' + str(j))
