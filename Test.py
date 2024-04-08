import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from PIL import Image

pics_path = r'C:\Users\shawn\Documents\Kaggle2\test_processed'
kaggle_path = r'C:\Users\shawn\Documents\Kaggle2'
# processed_path = r'C:\Users\shawn\Documents\Kaggle\train_sorted_color'
# validation_path = r'C:\Users\shawn\Documents\Kaggle\validation_color'

# import face recognition tool
os.chdir(kaggle_path)
face_cascade = cv2.CascadeClassifier('face_rec.xml')


# Function for getting celeb ID (official list is not alphabetical)
os.chdir(kaggle_path)
allCategories = pd.read_csv('category.csv')
categories = allCategories['Category'].tolist()
alphabeticalNames = sorted(categories)
trainData = pd.read_csv('train.csv')
allFileNames = (trainData['File Name']).tolist()

def getCelebID(pic):                                              # returns the celebrity's name (and ID#) from 'XXXXX.png'
    celebName = trainData['Category'][allFileNames.index(pic)]
    celebID = categories.index(celebName)
    return [celebName, celebID]

def alphabeticalIDtoListID(alphabeticalID):                                             
    celebName = alphabeticalNames[alphabeticalID]
    row = categories.index(celebName)
    return row


# load table which has each pic name and matching celeb name
os.chdir(kaggle_path)
#model = keras.models.load_model('NN78.keras')
model = tf.keras.models.load_model('NN78x.keras')

os.chdir(kaggle_path)
allCategories = pd.read_csv('category.csv')
categories = allCategories['Category'].tolist()

os.chdir(pics_path)
pics = os.listdir() 
pics = sorted(pics,key=lambda x: int(os.path.splitext(x)[0]))
i = 0 # number of pictures processed for training
guess = []
idx = 292
for pic in pics:
    os.chdir(pics_path)
    
    img = tf.keras.preprocessing.image.load_img(
    pic, target_size=(160, 160)
)

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    x_test =img_array/255.
    
    pred = model.predict(x_test) 
    pred = np.argmax(pred)
    guess.append(categories[alphabeticalIDtoListID(pred)])
    i += 1
    if (i % 1000 == 0):
        print(i)
    

import csv

os.chdir(kaggle_path)
with open('submission5.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["Id", "Category"]
    
    writer.writerow(field)
    for i in range(len(guess)):
        writer.writerow([str(i),str(guess[i])])
