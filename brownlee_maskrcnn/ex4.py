import copy
import json
import time
from os import listdir
# example of inference with a pre-trained coco model
from os.path import isfile, join

import numpy as np
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import save_image
from scipy.spatial import distance

rcnn = None


class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80


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

SPORT_BALL = 'sports ball'
TENNIS_RACKET = 'tennis racket'
PERSON = 'person'
BASEBALL_BAT = 'baseball bat'

# box:  y1, x1, y2, x2
Y1 = 0
X1 = 1
Y2 = 2
X2 = 3


def readJsonFile(filePath):
    with open(filePath) as f:
        data = json.load(f)
        return data


def get19hifiPosesFile(frameNumber):
    return f'/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/out/stanford_19hifi/stanford_19sec_clip_hifi_000000000{str(frameNumber - 1).zfill(3)}_keypoints.json'


def getSingleSwingPosesFile(frameNumber):
    return f'/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/out/stanford_single_swing/stanford_single_swing_000000000{str(frameNumber - 1).zfill(3)}_keypoints.json'


def getSingleBackhandPosesFile(frameNumber):
    return f'/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/out/stanford_single_backhand/{frameNumber}.json'


# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
m_i_bodyPart = {0: "Nose",
                1: "Neck",
                2: "RShoulder",
                3: "RElbow",
                4: "RWrist",
                5: "LShoulder",
                6: "LElbow",
                7: "LWrist",
                8: "MidHip",
                9: "RHip",
                10: "RKnee",
                11: "RAnkle",
                12: "LHip",
                13: "LKnee",
                14: "LAnkle",
                15: "REye",
                16: "LEye",
                17: "REar",
                18: "LEar",
                19: "LBigToe",
                20: "LSmallToe",
                21: "LHeel",
                22: "RBigToe",
                23: "RSmallToe",
                24: "RHeel"}
m_bodyPart_i = dict((v, k) for k, v in m_i_bodyPart.items())

neckPartNumber = m_bodyPart_i['Neck']
midHipPartNumber = m_bodyPart_i['MidHip']
rightWristNumber = m_bodyPart_i['RWrist']
rightElbowNumber = m_bodyPart_i['RElbow']
rightShoulderNumber = m_bodyPart_i['RShoulder']


def getReadablePose(pose):
    readablePose = {}
    for i, part in m_i_bodyPart.items():
        x, y = getBodyPartCoordinates(i, pose)
        score = getBodyPartScore(i, pose)
        readablePose[part] = f'({x}, {y})  {score}'
    return readablePose


def getBodyPartCoordinates(bodyPartNumber, pose):
    keyPoints = pose['pose_keypoints_2d']
    return keyPoints[bodyPartNumber * 3 + 0], keyPoints[bodyPartNumber * 3 + 1]


def getBodyPartScore(bodyPartNumber, pose):
    keyPoints = pose['pose_keypoints_2d']
    return keyPoints[bodyPartNumber * 3 + 2]


def getPosePersonSize(person):
    '''from neck to hip'''
    # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
    neckCoordinates = getBodyPartCoordinates(neckPartNumber, person)
    midHipCoordinates = getBodyPartCoordinates(midHipPartNumber, person)
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
            mrcnnPersonIter = {'roi': boxes[i], 'mask': masks[:, :, i], 'score': scores[i]}
            mrcnnPersonSizeIter = getMrcnnPersonSize(mrcnnPersonIter)
            print(count, "- size:", mrcnnPersonSizeIter)
            if mrcnnPersonSizeIter > mrcnnPersonSize:
                mrcnnPersonSize = mrcnnPersonSizeIter
                mrcnnPerson = mrcnnPersonIter
    print("mrcnn size in getBiggestMrcnnPerson", getMrcnnPersonSize(mrcnnPerson))
    return mrcnnPerson


def getRightHandSide(pose):
    midHipCoordinates = getBodyPartCoordinates(midHipPartNumber, pose)
    rWristCoord = getBodyPartCoordinates(rightWristNumber, pose)
    if not midHipCoordinates[0] > 0:
        raise Exception('cant estimate mid hip location')
    if rWristCoord[0] > 0:
        if rWristCoord[0] > midHipCoordinates[0]:
            return 'right'
        else:
            return 'left'
    return None


def getTennisRacketSide(racketBox, personBox):
    if racketBox is not None and personBox is not None:
        # racketMrcnn = getBestFromMrcnn(TENNIS_RACKET, r)
        # racketBox = racketMrcnn['roi']
        personAveX = np.mean([personBox[X1], personBox[X2]])
        racketAveX = np.mean([racketBox[X1], racketBox[X2]])
        if racketAveX > personAveX:
            return 'right'
        else:
            return 'left'
    else:
        return None


# get tennis racket box


def getRightElbowSide(pose):
    midHipCoordinates = getBodyPartCoordinates(midHipPartNumber, pose)
    rElbowCoord = getBodyPartCoordinates(rightElbowNumber, pose)
    if not midHipCoordinates[0] > 0:
        raise Exception('cant estimate mid hip location')
    if rElbowCoord[0] > 0:
        if rElbowCoord[0] > midHipCoordinates[0]:
            return 'right'
        else:
            return 'left'
    return None


def getHeight(box):
    return box[2] - box[0]


def getWidth(box):
    return box[3] - box[1]


def cropImage(img, box):
    # box:  y1, x1, y2, x2
    # https://stackoverflow.com/a/41909861/8870055
    # img is ndarray [rows, columns, rgb]
    # img.shape. It returns a tuple of number of rows, columns and channels (if image is color): https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/py_basic_ops.html
    imgCropped = img.copy()[box[Y1]:box[Y2], box[X1]:box[X2], :]
    return imgCropped


def getBestFromMrcnn(objectName, r):
    boxes = r['rois']  # y1, x1, y2, x2  #https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
    masks = r['masks']
    class_ids = r['class_ids']
    scores = r['scores']
    class_namez = [class_names[class_id] for class_id in class_ids]
    mrcnnObject = {}
    mrcnnObjectScore = 0
    count = 0
    for i in range(len(boxes)):
        className = class_namez[i]
        if className == objectName:
            count += 1
            mrcnnObjectIter = {'roi': boxes[i], 'mask': masks[:, :, i], 'score': scores[i]}
            mrcnnObjectScoreIter = scores[i]
            print(count, objectName, "- score:", mrcnnObjectScoreIter)
            if mrcnnObjectScoreIter > mrcnnObjectScore:
                mrcnnObjectScore = mrcnnObjectScoreIter
                mrcnnObject = mrcnnObjectIter
    print("mrcnn score in getBestFromMrcnn", mrcnnObject['score'] if 'score' in mrcnnObject else None)
    return mrcnnObject


def normalizeMrcnnObject(object, cropBox, origHeight, origWidth):
    # just add x1 and y1 from cropBox to each object coordinate?
    x1 = cropBox[X1]
    y1 = cropBox[Y1]
    x2 = cropBox[X2]
    y2 = cropBox[Y2]
    boxWidth = getWidth(cropBox)
    norm = copy.deepcopy(object)  # works
    roi = norm['roi']
    score = norm['score']
    mask = norm['mask']
    roi[X1] = roi[X1] + x1
    roi[X2] = roi[X2] + x1
    roi[Y1] = roi[Y1] + y1
    roi[Y2] = roi[Y2] + y1
    # num rows and columns needed before and after to expand to original size:
    numRowsBefore = y1
    numColsBefore = x1
    numRowsAfter = origHeight - y2
    numColsAfter = origWidth - x2
    #
    # rows to add
    rowsBefore = np.zeros((numRowsBefore, boxWidth), dtype=bool)
    rowsAfter = np.zeros((numRowsAfter, boxWidth), dtype=bool)
    #
    # add the rows
    mask = np.concatenate([rowsBefore, mask], axis=0)
    mask = np.concatenate([mask, rowsAfter], axis=0)
    #
    # cols to add
    colsBefore = np.zeros((origHeight, numColsBefore), dtype=bool)
    colsAfter = np.zeros((origHeight, numColsAfter), dtype=bool)
    #
    # add the cols
    mask = np.concatenate([colsBefore, mask], axis=1)
    mask = np.concatenate([mask, colsAfter], axis=1)
    #
    norm['score'] = score
    norm['roi'] = roi
    norm['mask'] = mask
    return norm


def addObjectToMrcnnResult(r, normalizedObject, class_id):
    r['scores'] = np.append(r['scores'], normalizedObject['score'])
    r['rois'] = np.concatenate([r['rois'], [normalizedObject['roi']]], axis=0)
    r['class_ids'] = np.append(r['class_ids'], class_id)
    # newMask shape is (rows, cols)
    # we need it to be (rows, cols, 1) # https://stackoverflow.com/a/7372678/8870055
    newMask = normalizedObject['mask'][..., np.newaxis]
    r['masks'] = np.concatenate([r['masks'], newMask], axis=2)
    return r


def zoomToFindStuffHandler(img, r, personPose, mrcnnPerson):
    rz = copy.deepcopy(r)
    foundClassNames = [class_names[class_id] for class_id in r['class_ids']]
    #
    for i in range(4):
        if SPORT_BALL not in foundClassNames or (TENNIS_RACKET not in foundClassNames and BASEBALL_BAT not in foundClassNames):
            if SPORT_BALL not in foundClassNames:
                print(i, 'missing ' + SPORT_BALL)
            if TENNIS_RACKET not in foundClassNames and BASEBALL_BAT not in foundClassNames:
                print(i, 'missing ' + TENNIS_RACKET)
            rz = zoomToFindStuff(img, rz, personPose, mrcnnPerson, i)
            # display_instances(img, rz['rois'], rz['masks'], rz['class_ids'], class_names, rz['scores'])
            foundClassNames = [class_names[class_id] for class_id in rz['class_ids']]
        else:
            break
    return rz


def getActionSide(personBox, personPose, racketBox):
    rightHandSide = getRightHandSide(personPose)

    tennisRacketSide = getTennisRacketSide(racketBox, personBox)
    rightElbowSide = getRightElbowSide(personPose)
    side = rightHandSide or tennisRacketSide or rightElbowSide
    return side


def zoomToFindStuff(img, r, personPose, mrcnnPerson, zoomNumber):
    """
    :param img: numpy.ndarray
    :param r: rcnn.detect()[0]
    :return: r with newly detected objects
    """
    rz = copy.deepcopy(r)
    #
    class_namez = [class_names[class_id] for class_id in rz['class_ids']]
    personBox = mrcnnPerson['roi']  # y1, x1, y2, x2
    # 1. determine if player right hand is to his left or right
    #
    ballFound = SPORT_BALL in class_namez
    racketFound = TENNIS_RACKET in class_namez or BASEBALL_BAT in class_namez
    #
    # 2. expand crop to player box plus left or right
    #
    cropBox = copy.deepcopy(personBox)
    #
    racketMrcnn = getBestFromMrcnn(TENNIS_RACKET, r)
    racketBox = racketMrcnn['roi'] if 'roi' in racketMrcnn else None
    side = getActionSide(personBox, personPose, racketBox)
    if zoomNumber == 0:
        cropBox[Y1] = max(cropBox[Y1] - getHeight(personBox) * 0.3, 0)
        if side == 'right':
            cropBox[X2] = min(cropBox[X2] + getWidth(personBox), img.shape[1])
        else:
            cropBox[X1] = max(cropBox[X1] - getWidth(personBox), 0)
    elif zoomNumber == 1:
        cropBox[Y1] = max(cropBox[Y1] - getHeight(personBox) * 0.5, 0)
        if side == 'right':
            cropBox[X2] = min(cropBox[X2] + getWidth(personBox) * 1.5, img.shape[1])
        else:
            cropBox[X1] = max(cropBox[X1] - getWidth(personBox) * 1.5, 0)
    elif zoomNumber == 2:
        cropBox[Y1] = max(cropBox[Y1] - getHeight(personBox) * 1.5, 0)
        cropBox[X2] = min(cropBox[X2] + getWidth(personBox) * 1.5, img.shape[1])
        cropBox[X1] = max(cropBox[X1] - getWidth(personBox) * 1.5, 0)
    elif zoomNumber == 3:
        cropBox[Y1] = max(cropBox[Y1] - getHeight(personBox) * 1.5, 0)
        cropBox[X2] = min(cropBox[X2] + img.shape[2] * 0.7, img.shape[1])
        cropBox[X1] = max(cropBox[X1] - img.shape[2] * 0.7, 0)
    else:
        raise Exception("invalid zoom number")
    #
    # 3. actually crop the image now
    #
    imgCropped = cropImage(img, cropBox)
    #
    # 4.  run maskrcnn
    #
    r2 = rcnn.detect([imgCropped], verbose=0)[0]
    # display_instances(imgCropped, r2['rois'], r2['masks'], r2['class_ids'], class_names, r2['scores'])
    #
    class_namez2 = [class_names[class_id] for class_id in r2['class_ids']]
    #
    if not ballFound and SPORT_BALL in class_namez2:
        print('found ball!!!!!')
        # now convert ball result (box and mask) to original image coordinate
        ball = getBestFromMrcnn(SPORT_BALL, r2)
        normalizedObject = normalizeMrcnnObject(ball, cropBox, img.shape[0], img.shape[1])
        class_id = class_names.index(SPORT_BALL)
        rz = addObjectToMrcnnResult(rz, normalizedObject, class_id)
    #
    if not racketFound and (TENNIS_RACKET in class_namez2 or BASEBALL_BAT in class_namez2):
        print('found racket!!!!!')
        # now convert ball result (box and mask) to original image coordinate
        if TENNIS_RACKET in class_namez2:
            racket = getBestFromMrcnn(TENNIS_RACKET, r2)
        else:
            racket = getBestFromMrcnn(BASEBALL_BAT, r2)
        normalizedObject = normalizeMrcnnObject(racket, cropBox, img.shape[0], img.shape[1])
        class_id = class_names.index(TENNIS_RACKET)
        rz = addObjectToMrcnnResult(rz, normalizedObject, class_id)
        # display_instances(img, rz['rois'], rz['masks'], rz['class_ids'], class_names, rz['scores'])
    return rz


def getRawVideoFrames(videoName):
    inputDir = f'/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/{videoName}'
    inputFiles = [join(inputDir, f) for f in listdir(inputDir) if isfile(join(inputDir, f)) and '.png' in f]
    inputFiles.sort()
    return inputFiles


start = time.time()


def getRawImgPath(videoName, frameNumber):
    # f"{1:02d}"
    return f'/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/{videoName}/{frameNumber:06d}.png'


def getAnalysisPath(videoName):
    # f"{1:02d}"
    return f'/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/analysis/{videoName}.json'


def getPosesPath(videoName, frameNumber):
    return f'/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/pose/data/{videoName}/{frameNumber:06d}_keypoints.json'


def getMrcnnImgPath(videoName, frameNumber):
    # /Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/segment/data/{videoName}/
    return f'/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/segment/img/{videoName}/{frameNumber:06d}.png'


def getMrcnnDataPath(videoName, frameNumber):
    return f'/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/segment/data/{videoName}/{frameNumber:06d}.npy'


def qaPoses(videoName):
    '''ensure each main player pose has a neck and right elbow'''
    numFrames = len(getRawVideoFrames(videoName))
    for frameNumber in range(1, numFrames + 1):
        posesPath = getPosesPath(videoName, frameNumber)
        poses = readJsonFile(posesPath)
        pose = getMainPlayerPose(poses)
        assert getBodyPartCoordinates(neckPartNumber, pose)[0] > 0, "neckPartNumber frame: %r" % frameNumber
        assert getBodyPartCoordinates(neckPartNumber, pose)[1] > 0, "neckPartNumber frame: %r" % frameNumber
        assert getBodyPartCoordinates(midHipPartNumber, pose)[0] > 0, "midHipPartNumber frame: %r" % frameNumber
        assert getBodyPartCoordinates(midHipPartNumber, pose)[1] > 0, "midHipPartNumber frame: %r" % frameNumber
        assert getBodyPartCoordinates(rightElbowNumber, pose)[0] > 0, "rightElbowNumber frame: %r" % frameNumber
        assert getBodyPartCoordinates(rightElbowNumber, pose)[1] > 0, "rightElbowNumber frame: %r" % frameNumber
        # assert getBodyPartCoordinates(rightWristNumber, pose)[0] > 0, "rightWristNumber frame: %r" % frameNumber
        # assert getBodyPartCoordinates(rightWristNumber, pose)[1] > 0, "rightWristNumber frame: %r" % frameNumber


def getSegments(videoName):
    rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
    rcnn.load_weights('models/mask_rcnn_coco.h5', by_name=True)
    #
    # qaPoses(videoName)
    #
    start = time.time()
    numFrames = len(getRawVideoFrames(videoName))
    #
    for frameNumber in range(1, numFrames + 1):
        inputImgPath = getRawImgPath(videoName, frameNumber)
        posesPath = getPosesPath(videoName, frameNumber)
        outputImgPath = getMrcnnImgPath(videoName, frameNumber)
        outputDataPath = getMrcnnDataPath(videoName, frameNumber)
        # if frameNumber < 98:
        #     continue
        print("\n input image: ", inputImgPath, "\nposes:", posesPath)
        poses = readJsonFile(posesPath)
        personPose = getMainPlayerPose(poses)
        # pprint.pprint(getReadablePose(personPose))
        print("main person size", getPosePersonSize(personPose))
        #                                                                   r, img = iterativeMaskrcnn(imgPath, personPose)
        imgAr = img_to_array(load_img(inputImgPath))
        r = rcnn.detect([imgAr], verbose=0)[0]
        r0 = copy.deepcopy(r)
        mrcnnPerson = getBiggestMrcnnPerson(r)
        print("mrcnn person size", getMrcnnPersonSize(mrcnnPerson))
        #
        r = zoomToFindStuffHandler(imgAr, r, personPose, mrcnnPerson)
        #
        save_image(imgAr, outputImgPath.replace('.png', ''), r['rois'], r['masks'], r['class_ids'], r['scores'], class_names)
        np.save(outputDataPath, r)
        print("elapsed time: ", time.time() - start)
        # display_instances(imgAr, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        # break

# getSegments("19sec")

# display_instances(img, r3['rois'], r3['masks'], r3['class_ids'], class_names, r3['scores'])
# break

# display_instances(img, r0['rois'], r0['masks'], r0['class_ids'], class_names, r0['scores'])
#
# figure out why bounding box for main player is too wide to the right in final r.  it's showing the cropbox ...
# display_instances(img, r0['rois'], r0['masks'], r0['class_ids'], class_names, r0['scores'])
# save_image(img, output_name, r['rois'], r['masks'], r['class_ids'], r['scores'], class_names)
# np.save(output_name + '.npy', r)
# print("elapsed time: ", time.time() - start)
