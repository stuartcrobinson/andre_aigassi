import os

import cv2

import imgTools as tools
from cctf.cctfTools import getColor

"""
doing QA on I-No Liao's TrackNet's badminton tracking traing data from:
https://inoliao.github.io/CoachAI/
https://drive.google.com/uc?export=download&id=1ZgoGm5y3_fSwzWLBFe_4Zu4LnMMkUd0J

load coordinates csv
for each badminton frame
    load the image
    draw an open circle around the coordinates
    save frame to video
write video
"""

"""
this file displays each image in the training data with a green circle around the coordinates listed in Badminton_label.csv

purpose is to make sure the listed coordinates are actually effectively tracking the birdie.

also to make sure we're cropping and resizing correctly. ✓✓
"""

coordinatesFile = "/Users/stuartrobinson/repos/computervision/andre_aigassi/images/badminton/Badminton_dataset/Badminton_label.csv"
imagesDir = "/Users/stuartrobinson/repos/computervision/andre_aigassi/images/badminton/Badminton_dataset/video_frames"


def getImage(frameNumber):
    path = os.path.join(imagesDir, str(frameNumber) + '.jpg')
    # print("path", path)
    image = cv2.imread(path)
    return image


def loadAllLabeledBadmintonFramesInBlackAndWhite(imgHeight, left, right):
    # resizeFactor = 5
    with open(coordinatesFile) as f:
        next(f)
        for i, line in enumerate(f):
            if i < 15800:
                continue
            frameNumber, visibility, x, y = list(map(int, line.strip().split(',')))
            print(frameNumber, visibility, x, y)
            im = getImage(frameNumber)[:, left:right, :]
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, (imgHeight, imgHeight))


def getResizeFactor(targetHeight, image):
    # crop image to center square
    shape = image.shape
    width = shape[1]
    height = shape[0]
    centerColumn = width // 2
    halfHeight = height // 2
    left = centerColumn - halfHeight
    right = centerColumn + halfHeight
    image = image[:, left:right, :]
    resizeFactr = height / targetHeight
    return resizeFactr, left, right


import numpy as np


def getBadmintonCoordinatesAndConf(h):
    resizeFactor, left, right = getResizeFactor(h, getImage(1))

    n_frames = sum(1 for line in open(coordinatesFile)) - 1
    visAndCoordinates = np.zeros([n_frames, 3], dtype=int)
    #
    with open(coordinatesFile) as f:
        next(f)
        for i, line in enumerate(f):
            frameNumber, visibility, x, y = list(map(int, line.strip().split(',')))
            xAdjusted = int((max(x - left, 0)) / resizeFactor)
            yAdjusted = int(y / resizeFactor)
            visAndCoordinates[frameNumber - 1, 0] = visibility
            visAndCoordinates[frameNumber - 1, 1] = xAdjusted
            visAndCoordinates[frameNumber - 1, 2] = yAdjusted
    # visAndCoordinates = np.delete(visAndCoordinates, 0, 0)
    return visAndCoordinates


def runQA(targetHeight):
    numFrames = sum(1 for line in open(coordinatesFile)) - 1

    resizeFactor, left, right = getResizeFactor(targetHeight, getImage(1))

    visAndCoords = getBadmintonCoordinatesAndConf(targetHeight)
    print("visAndCoords.shape:", visAndCoords.shape)
    for i in range(1, visAndCoords.shape[0]):
        # if i < 400:
        #     continue
        frameNumber = i
        visibility, x, y = visAndCoords[i]  # list(map(int, line.strip().split(',')))
        print(frameNumber, visibility, x, y)
        im = getImage(frameNumber)[:, left:right, :]
        im = cv2.resize(im, (targetHeight, targetHeight))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (targetHeight*3, targetHeight*3))

        tools.writeTextTopLeft(im, str(frameNumber) + " of " + str(numFrames))
        cv2.circle(im, (x*3, y*3), 10, (0, 255, 0), thickness=2)
        cv2.imshow("asdf2", im)
        cv2.waitKey(100)


def runQaOnNpyCctfFrames(imgHeight):
    images = np.load(f'/Users/stuartrobinson/repos/computervision/andre_aigassi/cctf/badmintonProcessedFrames_full_{imgHeight}_safe.npy')
    visAndCoords = getBadmintonCoordinatesAndConf(imgHeight)

    black = getColor(images[0].shape, 0, 0, 0)
    white = getColor(images[0].shape, 255, 255, 255)
    for i in range(images.shape[0]):
        im = images[i]
        im = im * 10
        im = np.minimum(im, white)
        im = np.maximum(im, black)
        im = im.astype('uint8')
        im = cv2.resize(im, (imgHeight*3, imgHeight*3))
        vis, x, y = visAndCoords[i]
        # tools.writeTextTopLeft(im, str(i) + " of " + str(images.shape[0]))
        cv2.circle(im, (x*3, y*3), 10, (0, 255, 0), thickness=2)
        cv2.imshow("asdf2", im)
        cv2.waitKey(100)


# runQA(112)
# runQaOnNpyCctfFrames(112)

"""
result -- data looks great !!! thank you I-No Liao!!!!

TODO next ... build CNN ?  
"""
