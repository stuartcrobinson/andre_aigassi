"""
goal: beat https://inoliao.github.io/CoachAI/

define and run CNN on cctf images (cctf = color-coded temporal filter)

how?  read training data as frames.

resize frame

create generator to pull the next frame and feed into model

model (try fewer filters than vgg and alexnet.  see modelOutlines_alexnet_vgg16.py)
        rules of thumb: https://www.reddit.com/r/MachineLearning/comments/3l5qu7/rules_of_thumb_for_cnn_architectures/

Conv2D(16, (3, 3)
Conv2D(16, (3, 3)
MaxPooling2D((4, 4), strides=(2, 2)

Conv2D(16, (3, 3)
Conv2D(16, (3, 3)
MaxPooling2D((4, 4), strides=(2, 2)

Conv2D(16, (3, 3)
Conv2D(16, (3, 3)
MaxPooling2D((4, 4), strides=(2, 2)

classes=3 (x, y, confidence)
Dense(500,
Dense(500,
Dense(classes,

TODO find simple keras cnn examples.  ground up.
    https://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/
    https://github.com/adventuresinML/adventures-in-ml-code/blob/b8764623cae350409f13d31eb87f0022a17e2403/keras_cnn.py

https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
No activation function is used for the output layer because it is a regression problem and we are interested in predicting numerical values directly without transform.

just read all images into memory?  only 4gb full-size full color

"""

import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from cctf.cctfTools import getResizeFactor, getImage, coordinatesFile, getBadmintonCoordinatesAndConf


def generateTestAndTrain(images, visAndCoords, imgHeight):
    print("running cross_validate...")
    x_train, x_test, y_train, y_test = cross_validate(images, visAndCoords)
    # convert the data to the right type
    # print("convert the data to the right type...")
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    print("save .npy files...")
    np.save(f'x_train_{imgHeight}.npy', x_train)
    np.save(f'x_test_{imgHeight}.npy', x_test)
    np.save(f'y_train_{imgHeight}.npy', y_train)
    np.save(f'y_test_{imgHeight}.npy', y_test)
    return x_train, x_test, y_train, y_test


def loadTestAndTrain(imgHeight):
    print("loadTestAndTrain...")
    x_train = np.load(f'x_train_{imgHeight}.npy')
    x_test = np.load(f'x_test_{imgHeight}.npy')
    y_train = np.load(f'y_train_{imgHeight}.npy')
    y_test = np.load(f'y_test_{imgHeight}.npy')
    # convert the data to the right type
    print("convert the data to the right type...")
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return x_train, x_test, y_train, y_test


def cross_validate(Xs, Ys):
    X_train, X_test, Y_train, Y_test = train_test_split(Xs, Ys, test_size=0.2, random_state=0)
    return X_train, X_test, Y_train, Y_test


batch_size = 128
num_classes = 3
epochs = 10

# imgHeight = 224
imgHeight = 112
# input image dimensions
img_x, img_y = imgHeight, imgHeight

# load all labeled images into a list of ndarrays
resizeFactor, left, right = getResizeFactor(imgHeight, getImage(1))

n_frames = sum(1 for line in open(coordinatesFile)) - 1

print("load images size " + str(imgHeight))
images = np.load(f'/Users/stuartrobinson/repos/computervision/andre_aigassi/cctf/badmintonProcessedFrames_full_{imgHeight}_safe.npy')
visAndCoords = getBadmintonCoordinatesAndConf(imgHeight)

x_train, x_test, y_train, y_test = generateTestAndTrain(images, visAndCoords, imgHeight)
x_train, x_test, y_train, y_test = loadTestAndTrain(imgHeight)

print("images.shape:", images.shape)
print("visAndCoords.shape:", visAndCoords.shape)
print("x_train.shape:", x_train.shape)
print("x_test.shape:", x_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)

# quit()
#

model = Sequential()
model.add(Conv2D(9, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(img_x, img_y, 3)))
model.add(Conv2D(9, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(18, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(Conv2D(18, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(18, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(4, 4)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

          # callbacks=[history])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
