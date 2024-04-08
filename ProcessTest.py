# This script takes the large testing set, finds faces, crops to those faces, and resizes the image.

import cv2
import numpy as np
import os
from PIL import Image

# Define folder paths
pics_path = r'C:\Users\shawn\Documents\Kaggle2\test_raw' #\test            # large folder with all (color) training data
kaggle_path = r'C:\Users\shawn\Documents\Kaggle2'                 # top level folder
processed_path = r'C:\Users\shawn\Documents\Kaggle2\test_processed' #\test_color# folder to put processed training data in
    
# import face recognition tool
os.chdir(kaggle_path)
face_cascade = cv2.CascadeClassifier('face_rec.xml')

def cropToFace(pic): # Returns the cropped image if any faces are detected
    os.chdir(pics_path)
    im = Image.open(pic).convert('RGB')
    img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    layers = 3
    scale = 1.05
    width = np.min(img.shape[0:2])
    minWidth = width
    faces = face_cascade.detectMultiScale(img, scale, layers, minSize = (minWidth, minWidth))
    
    # try REALLY hard to find a face O-><-O
    while len(faces) < 1 and minWidth > width//4 and scale > 1.01:
        minWidth -= 20
        faces = face_cascade.detectMultiScale(img, scale, layers, minSize = (minWidth, minWidth))
    width = np.min(img.shape)
    minWidth = width//8
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
numPics = len(pics)
i = 0 
j = 0
for pic in pics:
    if (i % 100) == 0:
        print(str(100*round(i/numPics,2)) + '%, processed = ' + str(i))    
    [faceFound, img] = cropToFace(pic)
    if not faceFound:
        j += 1
    os.chdir(processed_path)
    i += 1
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #img = np.array(img)
    cv2.imwrite(pic, img) # save the image into the processed folder
    #plt.figure()
    #plt.imshow(img)
    
print('procesed = ' + str(i), ' No face in '+ str(j))    
    
