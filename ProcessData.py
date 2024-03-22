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
