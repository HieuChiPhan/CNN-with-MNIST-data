# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
X = dataset.iloc[:, 1:].values
XX_test = dataset_test.iloc[:, 0:].values
y = dataset.iloc[:, 0].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X_1 = LabelEncoder()
#X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#labelencoder_X_2 = LabelEncoder()
#X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [0])
Y = onehotencoder.fit_transform(y.reshape(-1, 1)).toarray()
#X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
X_train=X
y_train=Y.astype("int")

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
XX_test = sc.transform(XX_test)

X_train = X_train.reshape(-1, 28, 28, 1).astype("float32")
XX_test = XX_test.reshape(-1, 28, 28, 1).astype("float32")


# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense,BatchNormalization,Dropout
from keras.callbacks import LearningRateScheduler

classifier = Sequential()
 
classifier.add(Convolution2D(filters = 16, kernel_size = (3, 3), activation='relu', input_shape = (28, 28, 1)))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(filters = 16, kernel_size = (3, 3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(strides=(2,2)))
classifier.add(Dropout(0.25))

classifier.add(Convolution2D(filters = 16, kernel_size = (3, 3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(filters = 16, kernel_size = (3, 3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(strides=(2,2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())
classifier.add(Dense(1024, activation='relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(1024, activation='relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(1024, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(10, activation='softmax'))


classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

hist = classifier.fit_generator(datagen.flow(X_train, y_train, batch_size=16),
                           steps_per_epoch=500,
                           epochs=100, #Increase this when not on Kaggle kernel
                           verbose=2,  #1 for ETA, 0 for silent
#                           validation_data=(X_test[:400,:], y_val[:400,:]))#, #For speed
                           callbacks=[annealer])

y_pred = classifier.predict(XX_test)
y_final = np.argmax(y_pred, axis=1)
im_ID=np.array(range(1,28001))

# Part 3 - Writing file
submission=pd.DataFrame({'ImageId':im_ID,'Label':y_final})
submission=submission.set_index('ImageId')
print(submission)
submission.to_csv('submission.csv')




