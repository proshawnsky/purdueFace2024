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
#from tensorflow.keras.preprocessing.image import ImageDataGenerator, image_dataset_from_directory

# Define folder paths
pics_path = r'C:\Users\shawn\Documents\Kaggle\train_small'
kaggle_path = r'C:\Users\shawn\Documents\Kaggle'
processed_path = r'C:\Users\shawn\Documents\Kaggle\train_sorted'
validation_path = r'C:\Users\shawn\Documents\Kaggle\validation'

train_data_dir = processed_path
validation_data_dir = validation_path
nb_train_samples = 3932
nb_validation_samples = 2034
epochs = 30
img_width = 64
img_height = 64
batch_size = 200

# Data generator
img_width, img_height = 64, 64

train_ds = tf.keras.utils.image_dataset_from_directory(
  processed_path,
  label_mode='int',
  color_mode="grayscale",
  labels = 'inferred',
#   validation_split=0.2,
#   subset="training",
  shuffle=True,
  seed=123,
  image_size=(64, 64),
  batch_size=100)
input_shape = (64, 64, 1)
print(train_ds)

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(5, 5), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        #layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
        #layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        # layers.Dropout(0.5),
        layers.Dense(100, activation="softmax"),
    ]
)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./ 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode="grayscale",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    color_mode="grayscale",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
