import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from google.colab import drive
import os
import matplotlib.pyplot as plt


drive.mount('/content/drive')
pics_path = '/content/drive/My Drive/ECE_50024_Kaggle/pics'
kaggle_path = '/content/drive/My Drive/ECE_50024_Kaggle'
cropped_path = '/content/drive/My Drive/ECE_50024_Kaggle/cropped'
#pics_path = r'C:\Users\shawn\Documents\Kaggle\pics'
#kaggle_path = r'C:\Users\shawn\Documents\Kaggle'
#cropped_path = r'C:\Users\shawn\Documents\Kaggle\cropped'

# import face recognition tool
os.chdir(kaggle_path)
face_cascade = cv2.CascadeClassifier('face_rec.xml')

# get file names and celeb names for pictures 
allCategories = pd.read_csv('category.csv')
categories = allCategories['Category'].tolist()
trainData = pd.read_csv('train_small.csv')
allFileNames = (trainData['File Name']).tolist()

# get list of available pictures
os.chdir(pics_path)
pics = os.listdir(pics_path)

# for each picture, ...

x_train = []
y_train = []
i = 0
for pic in pics:
    os.chdir(pics_path)
    img = cv2.imread(pic)                                         # load the picture into an array
    
    if img is not None:                                           # only continue if the picture was succesfully loaded
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                # convert the image to grayscale
      faces = face_cascade.detectMultiScale(gray, 1.2, 4)         # attempt to detect any face(s)

      if len(faces) > 0:                                          # if any faces were detected, ...
          x, y, w, h = faces[0,:]                                 # extract the position and size of the face
          cropped = gray[y:y+h, x:x+w]                            # store a cropped version of the image using the face rectangle
          cropped_size = 64                                      # set the size of all images
          resized = cv2.resize(cropped, (cropped_size, cropped_size)) # resize all images to be the same dimensions
          x_train.append(resized)
          
          celebName = trainData['Category'][allFileNames.index(pic)]
          celebID = categories.index(celebName)
          y_train.append(celebID)
          i += 1
          if (i%100 ==0):
              print(i)

x_train_temp = np.stack(np.array(x_train))/255
y_train_temp = np.stack(y_train)

idxToTrain = 700 # number of training samples
x_test = x_train_temp[idxToTrain:,:,:]
x_train = x_train_temp[:idxToTrain,:,:]
y_test = y_train_temp[idxToTrain:]
y_train = y_train_temp[:idxToTrain]
#__________________________________________________________________________________________________________________________________

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

#_______________________________________________________________________________________________________________________________

num_classes = 100
input_shape = (cropped_size, cropped_size, 1)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_validation = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(5, 5), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        #layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        #layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        # layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()

#__________________________________________________________________________________________________
batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

#score = model.evaluate(x_test, y_test, verbose=0)
#print("Test loss:", score[0])
#print("Test accuracy:", score[1])
import matplotlib.pyplot as plt

# plot the result
idx = 15
plt.figure()
plt.imshow(x_test[idx,:,:,0], cmap='gray')
# plt.figure()
yhat = model.predict(x_test[idx:(idx+1)])
# plt.bar(np.arange(100),yhat[0])
# plt.xlim([0,100])
# plt.title('Prediction probability')
# plt.grid()
sorted(zip(yhat[0], categories), reverse=True)[:3]

len(x_test)
max_test=np.zeros(y_test.shape)
for test in range(len(x_test)):
  yhat = model.predict(x_test[test:(test+1)])
  temp = yhat[0]
  max_test[test] = max(temp)

plt.figure()
plt.plot(range(len(x_test)), max_test)
plt.xlim((0,100))
