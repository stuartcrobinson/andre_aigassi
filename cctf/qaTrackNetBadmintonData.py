import os

import cv2

import imgTools as tools
from cctf.cctfTools import getColorBlock

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


import numpy as np


# runQA(112)
# runQaOnNpyCctfFrames(112)

"""
result -- data looks great !!! thank you I-No Liao!!!!

TODO next ... build CNN ?  
"""
