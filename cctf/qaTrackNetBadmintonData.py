import os

import cv2

import imgTools as tools

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

purpose is to make sure the listed coordinates are actually effectively tracking the birdie
"""

coordinatesFile = "/Users/stuartrobinson/repos/computervision/andre_aigassi/images/badminton/Badminton_dataset/Badminton_label.csv"
imagesDir = "/Users/stuartrobinson/repos/computervision/andre_aigassi/images/badminton/Badminton_dataset/video_frames"

numFrames = sum(1 for line in open(coordinatesFile)) - 1


def getImage(frameNumber):
    path = os.path.join(imagesDir, str(frameNumber) + '.jpg')
    print("path", path)
    image = cv2.imread(path)
    tools.writeTextTopLeft(image, str(frameNumber) + " of " + str(numFrames))
    return image


with open(coordinatesFile) as f:
    next(f)
    for i, line in enumerate(f):
        if i < 17800:
            continue
        frameNumber, visibility, x, y = list(map(int, line.strip().split(',')))
        print(frameNumber, visibility, x, y)
        im = getImage(frameNumber)
        print(type(x))
        print(type(y))
        # cv2.circle(im, (y, x), 10, (0,255,0), thickness=2, lineType=8, shift=0)
        cv2.circle(im, (x, y), 10, (0, 255, 0), thickness=2)
        cv2.imshow("asdf2", im)
        cv2.waitKey(10)

"""
result -- data looks great !!! thank you I-No Liao!!!!
"""
