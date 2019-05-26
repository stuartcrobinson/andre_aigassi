# v4.  cropping to look for sportball and tennis racket (or baseball bat)

#
# TODO - for each frame:
# load pose data
# if there's not a tennis racket and a sport ball identified:
#       find largest person box
#           make sure this matches largest person pose coordinates
#       crop image to only person box plus his width left of right - depending on where his right hand is (use pose data)
#       keep expanding crop until ball and racket are found, or it's pretty big
#       save segmentation data:
#           1.  bounding box for sport ball
#           2.  tennis racket polygons
#           3.  tennis racket:
#                   - farthest coordinate from player shoulder center
#                   - nearest coordinate from player shoulder center
#                   ---> to get specific points to connect a line between racket positions to make a nice curve hopefully??
#           4.  player polygons
#                   - this is so the tennis swing blur doesn't cover up the player's body
#
#
#


# example of inference with a pre-trained coco model
from os.path import isfile, join
from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances
from mrcnn.visualize import save_image
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import numpy as np
import json

import time
from scipy.spatial import distance

start = time.time()

# define 81 classes that the coco model knowns about
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# define the test configurationÒ


class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80


# define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
# load coco model weights
rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)

# load photograph
# imgName = '/Users/stuartrobinson/repos/computervision/attempt1/img/in/bball1.png'

# get all image files in:  /Users/stuartrobinson/repos/computervision/attempt1/tennis_video/frames/stanford_single_backhand
# taht start with "thumb"

# inputDir = '/Users/stuartrobinson/repos/computervision/attempt1/tennis_video/frames/stanford_single_backhand'
inputDir = '/Users/stuartrobinson/repos/computervision/attempt1/tennis_video/frames/stanford_single_swing'
outputDir = '/Users/stuartrobinson/repos/computervision/attempt1/tennis_video/frames/stanford_single_swing_segmented'

inputFiles = [f for f in listdir(inputDir) if isfile(join(inputDir, f)) and 'thumb' in f]

start = time.time()


# /Users/stuartrobinson/repos/computervision/attempt1/tennis_video/frames/stanford_single_swing/thumb00001.png

def getPose(frameNumber):
    path = f'/Users/stuartrobinson/repos/computervision/attempt1/tennis_video/out/stanford_single_swing/stanford_single_swing_000000000{str(frameNumber).zfill(3)}_keypoints.json'
    print(path)
    with open(path) as f:
        data = json.load(f)
        return data


def getPosePersonSize(person):
    # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
    neckPartNumber = 1
    midHipPartNumber = 8
    keyPoints = person['pose_keypoints_2d']
    neckCoordinates = (keyPoints[neckPartNumber * 3 + 0], keyPoints[neckPartNumber * 3 + 1])
    midHipCoordinates = (keyPoints[midHipPartNumber * 3 + 0], keyPoints[midHipPartNumber * 3 + 1])
    if neckCoordinates[0] == 0 or midHipCoordinates[0] == 0:
        return -1
    return distance.euclidean(neckCoordinates, midHipCoordinates)


def getMainPlayerPose(poses):
    people = poses['people']
    biggestPerson = {}
    biggestSize = 0
    count = 0
    for person in people:
        size = getPosePersonSize(person)
        count += 1
        print(count, "- size:", size)
        if size > biggestSize:
            biggestSize = size
            biggestPerson = person
    return biggestPerson


def getMrcnnPersonSize(p):
    # y1, x1, y2, x2  #https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
    roi = p['roi']
    # find the first row (starts at the top) that contains a true value
    # find the last row that ocntains a true value
    return roi[2] - roi[0]


def getBiggestMrcnnPerson(r):
    boxes = r['rois']  # y1, x1, y2, x2  #https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
    masks = r['masks']
    class_ids = r['class_ids']
    scores = r['scores']
    class_namez = [class_names[class_id] for class_id in class_ids]
    mrcnnPerson = {}
    mrcnnPersonSize = 0
    count = 0
    for i in range(len(boxes)):
        className = class_namez[i]
        if className == 'person':
            count += 1
            mrcnnPersonIter = {'roi': boxes[i], 'mask': masks[i], 'score': scores[i]}
            mrcnnPersonSizeIter = getMrcnnPersonSize(mrcnnPersonIter)
            print(count, "- size:", mrcnnPersonSizeIter)
            if mrcnnPersonSizeIter > mrcnnPersonSize:
                mrcnnPersonSize = mrcnnPersonSizeIter
                mrcnnPerson = mrcnnPersonIter
    print("mrcnn size in getBiggestMrcnnPerson", getMrcnnPersonSize(mrcnnPerson))
    return mrcnnPerson


for imgName in inputFiles:
    frameNumber = int(imgName.split('thumb')[1].split('.')[0])
    imgPath = join(inputDir, imgName)
    print("processing: ", frameNumber, imgPath)
    poses = getPose(frameNumber - 1)  # for single swing only.  i stupidly renamed them for the single backhand
    # print(pose)
    print(len(poses['people']), "people")
    personPose = getMainPlayerPose(poses)
    print("size", getPosePersonSize(personPose))
    # continue
    img = load_img(imgPath)
    img = img_to_array(img)
    # make prediction
    results = rcnn.detect([img], verbose=0)
    # get dictionary for first prediction
    r = results[0]
    # show photo with bounding boxes, masks, class labels and scores
    image = img
    boxes = r['rois']  # y1, x1, y2, x2  #https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
    masks = r['masks']
    class_ids = r['class_ids']
    scores = r['scores']
    class_namez = [class_names[class_id] for class_id in class_ids]
    mrcnnPerson = getBiggestMrcnnPerson(r)
    print("mrcnn size", getMrcnnPersonSize(mrcnnPerson))
    # continue
    if 'sport ball' not in class_namez or 'tennis racket' not in class_namez or 'baseball bal' not in class_namez:
        # crop image left or right depending on where right hand is.
        # cropping looks easy!!!! just take subset of img - from img_to_array
        # the dimensions of this numpy array match the image dimensions!  :)

        # first do smallest crop of player box plus box-size areas left and right (start with the side of right hand)
        # then keep expanding toward the center of the image.
        # then expand everything 25% north. then left and right. then 50% north

        # just tedious arithmetic to add newly-found objects to main photo size.
        # what if two different balls or rackets are found?  (without overlapping boxes?) keep the one closest to main player
        #   ADVANCED - keep the ball most in-line with ball coordinate in surrounding images.

        # and then save this stuff as numpy arrays.
        # later, we figure out how to trace tennis ball and racket on screen.  getting closest/farthest racket points from player shoulder etc

        # check out save_image for image-drawing tips
        # draw lightning pointing to ball hitting racket ? detect hit instant when ball is visible through the racket. or very close

        # the FIRST frame where ball is visible through racket

        # to find farthest racket and closest racket points, look in concentric circles radiating from shoulder point
        # how to draw curve from points?
        # https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
        # http://benpaulthurstonblog.blogspot.com/2013/07/smooth-curve-generator-implemented-in.html
        # not python: http://what-when-how.com/advanced-methods-in-computer-graphics/curves-and-surfaces-advanced-methods-in-computer-graphics-part-5/
        # spline?  https://gamedev.stackexchange.com/questions/38622/how-to-create-a-curve-from-a-set-of-points-that-passes-through-said-points
        # cubic spline? catmull-rom spline -- yes cubic!  found lots of resources
        #
        # drawing circles:
        # https://www.google.com/search?q=python+plot+circle+in+image+ndarray&oq=python+plot+circle+in+image+ndarray&aqs=chrome..69i57.10939j0j7&sourceid=chrome&ie=UTF-8
        # https://stackoverflow.com/questions/34902477/drawing-circles-on-image-with-matplotlib-and-numpy
        #
        # to get tennis ball actual width - find center of ball mask - and then find the closest border mask pixel.  to control for blur.
        pass
    output_name = join(outputDir, imgName)
    save_image(image, output_name, boxes, masks, class_ids, scores, class_names)
    np.save(output_name + '.npy', r)
    print("elapsed time: ", time.time() - start)
#     display_instances(image, boxes, masks, class_ids, class_names, scores)
