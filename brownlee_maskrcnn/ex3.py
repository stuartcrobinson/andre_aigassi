# example of inference with a pre-trained coco model
from os.path import isfile, join
from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances
from mrcnn.visualize import save_image #https://github.com/matterport/Mask_RCNN/pull/38 #https://github.com/matterport/Mask_RCNN/commit/bc8f148b820ebd45246ed358a120c99b09798d71
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import numpy as np

import time

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

for imgName in inputFiles:
    imgPath = join(inputDir, imgName)
    print("processing: ", imgPath)
    img = load_img(imgPath)
    img = img_to_array(img)
    # make prediction
    results = rcnn.detect([img], verbose=0)
    # get dictionary for first prediction
    r = results[0]
    # show photo with bounding boxes, masks, class labels and scores
    image = img
    boxes = r['rois']
    masks = r['masks']
    class_ids = r['class_ids']
    scores = r['scores']
    output_name = join(outputDir, imgName)
    save_image(image, output_name, boxes, masks, class_ids, scores, class_names)
    np.save(output_name + '.npy', r)
    print("elapsed time: ", time.time() - start)
#     display_instances(image, boxes, masks, class_ids, class_names, scores)

print("elapsed time: ", time.time() - start)
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

