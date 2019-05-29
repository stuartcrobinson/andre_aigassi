# this is cool http://pythontutor.com/visualize.html#mode=display  interactive visualizer
import copy
import math
import time

import pandas as pd
from scipy.spatial import distance

import brownlee_maskrcnn.ex4 as ex4

pd.set_option("display.max_rows", 600)
pd.set_option('display.float_format', lambda x: '%.1f' % x)

# box:  y1, x1, y2, x2
Y1 = 0
X1 = 1
Y2 = 2
X2 = 3


def getPlayerPoses(videoName):
    numFrames = len(ex4.getRawVideoFrames(videoName))
    playerPoses = []
    for frameNumber in range(1, numFrames + 1):
        posesPath = ex4.getPosesPath(videoName, frameNumber)
        poses = ex4.readJsonFile(posesPath)
        personPose = ex4.getMainPlayerPose(poses)
        playerPoses += [personPose]
    return playerPoses


def overlapsOtherBox(b, o):
    if b is None or o is None:
        return False
    topLeftIsInside_ = o[X1] < b[X1] < o[X2] and o[Y1] < b[Y1] < o[Y2]
    botRightIsInside = o[X1] < b[X2] < o[X2] and o[Y1] < b[Y2] < o[Y2]
    botLeftIsInside_ = o[X1] < b[X1] < o[X2] and o[Y1] < b[Y2] < o[Y2]

    return topLeftIsInside_ or botRightIsInside or botLeftIsInside_


def isInsideOtherBox(b, o):
    if b is None or o is None:
        return False
    topLeftIsInside_ = o[X1] < b[X1] < o[X2] and o[Y1] < b[Y1] < o[Y2]
    botRightIsInside = o[X1] < b[X2] < o[X2] and o[Y1] < b[Y2] < o[Y2]
    return topLeftIsInside_ and botRightIsInside


# add something here to make sure the proximal racket point is closer to his elbow than either foot
# no, ensure it's certain distance from wrist or elbow.  see frame 411 of 19sec
# to ensure this is the player's racket and not the opponen't racket
# to ensure it's not a shadow

def isValidRacket(racketBox, personBox, pose):
    if racketBox is None or personBox is None:
        return False
    elbowCoord = ex4.getBodyPartCoordinates(ex4.rightElbowNumber, pose)
    wristCoord = ex4.getBodyPartCoordinates(ex4.rightWristNumber, pose)
    racketBoxTL = (racketBox[X1], racketBox[Y1])
    racketBoxTR = (racketBox[X2], racketBox[Y1])
    racketBoxBR = (racketBox[X2], racketBox[Y2])
    racketBoxBL = (racketBox[X1], racketBox[Y2])
    if wristCoord[0] > 0:
        minDist = min(distance.euclidean(racketBoxTL, wristCoord),
                      distance.euclidean(racketBoxTR, wristCoord),
                      distance.euclidean(racketBoxBR, wristCoord),
                      distance.euclidean(racketBoxBL, wristCoord))
        if minDist > 70:
            print('wrist mindist', minDist)
            return False
    elif elbowCoord[0] > 0:
        minDist = min(distance.euclidean(racketBoxTL, elbowCoord),
                      distance.euclidean(racketBoxTR, elbowCoord),
                      distance.euclidean(racketBoxBR, elbowCoord),
                      distance.euclidean(racketBoxBL, elbowCoord))
        if minDist > 105:
            print('elbow mindist', minDist)
            return False
    # return True
    maxDimension = max(ex4.getWidth(racketBox), ex4.getHeight(racketBox))
    return maxDimension > 45
    # print('maxDimension', maxDimension)
    # return overlapsOtherBox(racketBox, personBox) or maxDimension > 60  # and maxDimension > 50)


def getStuffFromMrcnnNpyFiles(playerPoses):
    numFrames = len(ex4.getRawVideoFrames(videoName))
    personBoxes = [None for i in range(numFrames)]
    racketBoxes = [None for i in range(numFrames)]
    ballBoxes = [None for i in range(numFrames)]
    racketMasks = [None for i in range(numFrames)]
    for i in range(numFrames):
        # frameNumber = 346
        # i = 345
        frameNumber = i + 1
        print('frame: ', frameNumber)
        pose = playerPoses[i]
        r = np.load(ex4.getMrcnnDataPath(videoName, frameNumber)).item()
        personMrcnn = ex4.getBestFromMrcnn(ex4.PERSON, r)  # same as biggest i guess??
        racketMrcnn = ex4.getBestFromMrcnn(ex4.TENNIS_RACKET, r)
        ballMrcnn = ex4.getBestFromMrcnn(ex4.SPORT_BALL, r)
        personBox = personMrcnn['roi'] if 'roi' in personMrcnn else None
        racketBox = racketMrcnn['roi'] if 'roi' in racketMrcnn else None
        ballBox = ballMrcnn['roi'] if 'roi' in ballMrcnn else None
        personBoxes[i] = personBox
        if isValidRacket(racketBox, personBox, pose):
            racketBoxes[i] = racketBox
            racketMrcnn = ex4.getBestFromMrcnn(ex4.TENNIS_RACKET, r)
            racketMasks[i] = racketMrcnn['mask'] if 'mask' in racketMrcnn else None
        ballBoxes[i] = ballBox
    return personBoxes, racketBoxes, ballBoxes, racketMasks


def getRacketExtremeCoords(mask, rWristCoords, rElbowCoords, rShoulderCoords):
    # just walk through all mask coordinates that are true
    # how to get true coordinates w/out walking along False ones also??
    # len(np.where(m == True)[0])
    if mask is None:
        return None, None
    coords = np.where(mask == True)
    Y = coords[0]
    X = coords[1]
    reference = rWristCoords
    if reference[0] == 0:
        reference = rElbowCoords
    if reference[0] == 0:
        reference = rShoulderCoords
    if reference[0] == 0:
        return None, None
    proximal = (0, 0)
    proximalDistance = distance.euclidean(reference, proximal)
    distal = copy.deepcopy(reference)
    distalDistance = distance.euclidean(reference, distal)
    for i in range(len(Y)):
        y = Y[i]
        x = X[i]
        point = (x, y)
        pointDistance = distance.euclidean(reference, point)
        if pointDistance < proximalDistance:
            proximal = point
            proximalDistance = pointDistance
        if pointDistance > distalDistance:
            distal = point
            distalDistance = pointDistance
    return proximal, distal


def getRacketProxAndDist(x):
    i = x.name
    racketMask = racketMasks[i]
    pose = playerPoses[i]
    proximalCoords, distalCoords = getRacketExtremeCoords(racketMask,
                                                          ex4.getBodyPartCoordinates(ex4.rightWristNumber, pose),
                                                          ex4.getBodyPartCoordinates(ex4.rightElbowNumber, pose),
                                                          ex4.getBodyPartCoordinates(ex4.rightShoulderNumber, pose))
    return proximalCoords, distalCoords


def getrWristToFeetOver300(r, playerPoses):
    i = r.name
    pose = playerPoses[i]
    rWristCoords = ex4.getBodyPartCoordinates(ex4.rightWristNumber, pose)
    if rWristCoords[0] == 0:
        return False
    rFootCoords = ex4.getBodyPartCoordinates(ex4.rightFootNumber, pose)
    lFootCoords = ex4.getBodyPartCoordinates(ex4.leftFootNumber, pose)
    distRfoot = distance.euclidean(rWristCoords, rFootCoords)
    distLfoot = distance.euclidean(rWristCoords, lFootCoords)
    return distRfoot > 300 and distLfoot > 300


def getTbInR(x, tbBoxes, racketBoxes):
    i = x.name
    tbBox = tbBoxes[i]
    racketBox = racketBoxes[i]
    if isInsideOtherBox(tbBox, racketBox):
        return 2
    if overlapsOtherBox(tbBox, racketBox):
        return 1
    return 0


def distLam(x1, y1, x2, y2):
    if x1 is None or x2 is None or y1 is None or y2 is None:
        return None
    return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))


def getTennisBallRadiusAndCoordinates(x, tbBoxes):
    i = x.name
    tbBox = tbBoxes[i]
    # print(i, tbBox)
    if tbBox is None:
        return None, None, None
    width = ex4.getWidth(tbBox)
    height = ex4.getHeight(tbBox)
    radius = min(width, height) / 2
    coords = ex4.getCenter(tbBox)
    return radius, coords[0], coords[1]


videoName = "19sec"

numFrames = len(ex4.getRawVideoFrames(videoName))

playerPoses = getPlayerPoses(videoName)

# frameNumber = 346


playerBoxes, racketBoxes, tbBoxes, racketMasks = getStuffFromMrcnnNpyFiles(playerPoses)

analysis = ex4.readJsonFile(ex4.getAnalysisPath(videoName))

if len(analysis) < numFrames:
    analysis = [{} for i in range(numFrames)]

for i in range(numFrames):
    # print(i)
    pose = playerPoses[i]
    personBox = playerBoxes[i]
    racketBox = racketBoxes[i]
    side = ex4.getActionSide(personBox, pose, racketBox)
    analysis[i]['racket_side'] = side

df = pd.DataFrame(analysis)
df['f'] = df.index + 1

cols_to_order = ['f']
new_columns = cols_to_order + (df.columns.drop(cols_to_order).to_list())
df = df[new_columns]

start = time.time()
df['racket_proxAndDist'] = df.apply(lambda x: getRacketProxAndDist(x), axis=1)
print("elapsed time: ", time.time() - start)
# df = df[['racket_side','racket_proxAndDist']]

df[['racket_proximal', 'racket_distal']] = pd.DataFrame(df['racket_proxAndDist'].tolist(), index=df.index)

df['racket_proximal'] = df['racket_proximal'].apply(lambda x: (None, None) if x is None else x)
df['racket_distal'] = df['racket_distal'].apply(lambda x: (None, None) if x is None else x)

df[['rpx0', 'rpy0']] = pd.DataFrame(df['racket_proximal'].tolist(), index=df.index)
df[['rdx0', 'rdy0']] = pd.DataFrame(df['racket_distal'].tolist(), index=df.index)
df = df.drop(columns=['racket_proximal', 'racket_distal'])

df['tbInR'] = df.apply(lambda x: getTbInR(x, tbBoxes, racketBoxes), axis=1)

df['rdx1'] = df['rdx0'].shift(1)
df['rdy1'] = df['rdy0'].shift(1)
df['rdx2'] = df['rdx0'].shift(2)
df['rdy2'] = df['rdy0'].shift(2)
df['rdx3'] = df['rdx0'].shift(3)
df['rdy3'] = df['rdy0'].shift(3)
df['rdx4'] = df['rdx0'].shift(4)
df['rdy4'] = df['rdy0'].shift(4)
df['rdx5'] = df['rdx0'].shift(5)
df['rdy5'] = df['rdy0'].shift(5)

df['rdΔ1'] = df.apply(lambda x: distLam(x.rdx0, x.rdy0, x.rdx1, x.rdy1), axis=1)
df['rdΔ2'] = df.apply(lambda x: distLam(x.rdx0, x.rdy0, x.rdx2, x.rdy2), axis=1)
df['rdΔ3'] = df.apply(lambda x: distLam(x.rdx0, x.rdy0, x.rdx3, x.rdy3), axis=1)
df['rdΔ4'] = df.apply(lambda x: distLam(x.rdx0, x.rdy0, x.rdx4, x.rdy4), axis=1)
df['rdΔ5'] = df.apply(lambda x: distLam(x.rdx0, x.rdy0, x.rdx5, x.rdy5), axis=1)
df['rdΔ1μ6'] = df['rdΔ1'].rolling(6, min_periods=2).mean()

df['rdΔ1_est'] = df['rdΔ1']

for i in range(len(df['rdΔ1'])):
    # print(i)
    racketDistDelta1 = df['rdΔ1'].values[i]
    # print(i, racketDistDelta1)
    if pd.isna(racketDistDelta1):
        # print(True)
        racketDistDelta2 = df['rdΔ2'].values[i]
        if not pd.isna(racketDistDelta2):
            df['rdΔ1_est'].values[i] = racketDistDelta2 / 2
            if i > 0 and pd.isna(df['rdΔ1_est'].values[i - 1]):
                df['rdΔ1_est'].values[i - 1] = racketDistDelta2 / 2

for i in range(len(df['rdΔ1'])):
    # print(i)
    racketDistDelta1 = df['rdΔ1'].values[i]
    # print(i, racketDistDelta1)
    if pd.isna(racketDistDelta1):
        # print(True)
        racketDistDelta3 = df['rdΔ3'].values[i]
        if not pd.isna(racketDistDelta3):
            df['rdΔ1_est'].values[i] = racketDistDelta3 / 3
            if i > 0 and pd.isna(df['rdΔ1_est'].values[i - 1]):
                df['rdΔ1_est'].values[i - 1] = racketDistDelta3 / 3
            if i > 1 and pd.isna(df['rdΔ1_est'].values[i - 2]):
                df['rdΔ1_est'].values[i - 2] = racketDistDelta3 / 3

for i in range(len(df['rdΔ1'])):
    # print(i)
    racketDistDelta1 = df['rdΔ1'].values[i]
    # print(i, racketDistDelta1)
    if pd.isna(racketDistDelta1):
        # print(True)
        racketDistDelta4 = df['rdΔ4'].values[i]
        if not pd.isna(racketDistDelta4):
            df['rdΔ1_est'].values[i] = racketDistDelta4 / 4
            if i > 0 and pd.isna(df['rdΔ1_est'].values[i - 1]):
                df['rdΔ1_est'].values[i - 1] = racketDistDelta4 / 4
            if i > 1 and pd.isna(df['rdΔ1_est'].values[i - 2]):
                df['rdΔ1_est'].values[i - 2] = racketDistDelta4 / 4
            if i > 2 and pd.isna(df['rdΔ1_est'].values[i - 3]):
                df['rdΔ1_est'].values[i - 3] = racketDistDelta4 / 4

for i in range(len(df['rdΔ1'])):
    # print(i)
    racketDistDelta1 = df['rdΔ1'].values[i]
    # print(i, racketDistDelta1)
    if pd.isna(racketDistDelta1):
        # print(True)
        racketDistDelta5 = df['rdΔ5'].values[i]
        if not pd.isna(racketDistDelta5):
            df['rdΔ1_est'].values[i] = racketDistDelta5 / 5
            if i > 0 and pd.isna(df['rdΔ1_est'].values[i - 1]):
                df['rdΔ1_est'].values[i - 1] = racketDistDelta5 / 5
            if i > 1 and pd.isna(df['rdΔ1_est'].values[i - 2]):
                df['rdΔ1_est'].values[i - 2] = racketDistDelta5 / 5
            if i > 2 and pd.isna(df['rdΔ1_est'].values[i - 3]):
                df['rdΔ1_est'].values[i - 3] = racketDistDelta5 / 5
            if i > 3 and pd.isna(df['rdΔ1_est'].values[i - 4]):
                df['rdΔ1_est'].values[i - 4] = racketDistDelta5 / 5

df['rdΔ1μ4_est'] = df['rdΔ1_est'].rolling(4, min_periods=2).mean()

# df.query('tbInR > 0')

side6ago = df['racket_side'].shift(6)
df['sideswitched'] = df['racket_side'] != side6ago

df['rWristToFeetOver300'] = df.apply(lambda x: getrWristToFeetOver300(x, playerPoses), axis=1)
df['rWristToFeetOver300Recent'] = df['rWristToFeetOver300'] | df['rWristToFeetOver300'].shift(1) | df['rWristToFeetOver300'].shift(2) | \
                                  df['rWristToFeetOver300'].shift(3) | df['rWristToFeetOver300'].shift(4) | df['rWristToFeetOver300'].shift(5) | df['rWristToFeetOver300'].shift(6)


def encodeSwingType(r):
    if r['sideswitched'] == True and r['rdΔ1μ4_est'] > 50:
        if r['rWristToFeetOver300Recent']:
            return 'Serve'
        if r['racket_side'] == 'right':
            return 'Backhand'
        if r['racket_side'] == 'left':
            return 'Forehand'
    return None


df['swing'] = df.apply(lambda r: encodeSwingType(r), axis=1)

df = df.drop(errors='ignore',
             columns=['tbx', 'tby', 'rdx1', 'rdx2', 'rdx3', 'rdx4', 'rdx5', 'rdy2', 'rdy4', 'rpx1', 'rpx3', 'rpx5', 'rdy1', 'rdy3', 'rdy5', 'rpy1', 'rpy3', 'rpy5', 'tbRadiusAndCoords', 'racket_distal_delta_1', 'racket_distal_delta_3',
                      'racket_distal_delta_5', 'sideswitched', 'rdΔ1', 'rdΔ2', 'rdΔ3', 'rdΔ4', 'rdΔ5', 'rdΔ1μ6', 'rdΔ1_est', 'tbInR', 'rpx0',
                      'rpy0', 'rdx0', 'rdy0', 'rdΔ1μ4_est', 'rWristToFeetOver300', 'rWristToFeetOver300Recent', 'racket_proxAndDist'])
del side6ago

###################################################################################################################
###################################################################################################################
###################################################################################################################
# tennis ball tracking
###################################################################################################################
###################################################################################################################
###################################################################################################################

df = df[['f', 'racket_side', 'swing']]

df['bRadiusAndCoords'] = df.apply(lambda x: getTennisBallRadiusAndCoordinates(x, tbBoxes), axis=1)

df[['bRadius', 'bx', 'by']] = pd.DataFrame(df['bRadiusAndCoords'].tolist(), index=df.index)

df['bxe'] = df['bx']
df['bye'] = df['by']

for i in range(30, len(df) - 1):
    x = df['bxe'].values[i]
    y = df['bye'].values[i]
    xNext = df['bxe'].values[i + 1]
    yNext = df['bye'].values[i + 1]
    xPrev1 = df['bxe'].values[i - 1]
    yPrev1 = df['bye'].values[i - 1]
    xPrev2 = df['bxe'].values[i - 2]
    yPrev2 = df['bye'].values[i - 2]
    xPrev3 = df['bxe'].values[i - 3]
    yPrev3 = df['bye'].values[i - 3]
    xPrev4 = df['bxe'].values[i - 4]
    yPrev4 = df['bye'].values[i - 4]
    if pd.isna(x) and not pd.isna(xNext):
        if not pd.isna(xPrev1):
            df['bxe'].values[i] = (xPrev1 + xNext) / 2
            df['bye'].values[i] = (yPrev1 + yNext) / 2
        elif not pd.isna(xPrev2):
            x_lenPer = (xNext - xPrev2) / 3
            y_lenPer = (yNext - yPrev2) / 3
            df['bxe'].values[i] = xNext - x_lenPer
            df['bye'].values[i] = yNext - y_lenPer
            df['bxe'].values[i - 1] = xNext - 2 * x_lenPer
            df['bye'].values[i - 1] = yNext - 2 * y_lenPer
        elif not pd.isna(xPrev3):
            x_lenPer = (xNext - xPrev3) / 4
            y_lenPer = (yNext - yPrev3) / 4
            df['bxe'].values[i] = xNext - x_lenPer
            df['bye'].values[i] = yNext - y_lenPer
            df['bxe'].values[i - 1] = xNext - 2 * x_lenPer
            df['bye'].values[i - 1] = yNext - 2 * y_lenPer
            df['bxe'].values[i - 2] = xNext - 3 * x_lenPer
            df['bye'].values[i - 2] = yNext - 3 * y_lenPer
        elif not pd.isna(xPrev4):
            x_lenPer = (xNext - xPrev4) / 5
            y_lenPer = (yNext - yPrev4) / 5
            df['bxe'].values[i] = xNext - x_lenPer
            df['bye'].values[i] = yNext - y_lenPer
            df['bxe'].values[i - 1] = xNext - 2 * x_lenPer
            df['bye'].values[i - 1] = yNext - 2 * y_lenPer
            df['bxe'].values[i - 2] = xNext - 3 * x_lenPer
            df['bye'].values[i - 2] = yNext - 3 * y_lenPer
            df['bxe'].values[i - 3] = xNext - 4 * x_lenPer
            df['bye'].values[i - 3] = yNext - 4 * y_lenPer

df['isBallTrackStart'] = False
df['isBallTrackStart'].values[46] = True
df['isBallTrackStart'].values[106] = True
df['isBallTrackStart'].values[186] = True
df['isBallTrackStart'].values[262] = True
df['isBallTrackStart'].values[342] = True
df['isBallTrackStart'].values[413] = True
df['isBallTrackStart'].values[575] = True

################################################################################################################
################################################################################################################
################################################################################################################

################################################################################################################
################################################################################################################
################################################################################################################

# dont use expected coordinates for plotting the arc!  only to know how far out to draw it per frame

# lets start by just drawing line for the next 20 frames.
# make a map of key: start frame, value: arc

from scipy import interpolate

splines = {}

count = 0
for i in range(30, len(df) - 1):
    frameNumber = i + 1
    if df['isBallTrackStart'].values[i]:
        count += 1
        X = df['bx'].loc[i:i + 15].dropna().tolist()
        Y = df['by'].loc[i:i + 15].dropna().tolist()
        df2 = pd.DataFrame({'X': X, 'Y': Y})
        df2 = df2.sort_values('X')
        X_sorted = df2['X'].tolist()
        Y_sorted = df2['Y'].tolist()
        print()
        print(X)
        print(X_sorted)
        print()
        # print(X)
        # X_e = df['bxe'].loc[i:i + 20].dropna().tolist()
        # Y_e = df['bye'].loc[i:i + 20].dropna().tolist()
        splines[frameNumber] = {"tck": interpolate.splrep(X_sorted, Y_sorted), "X": X, "Y": Y}


# https://stackoverflow.com/a/31544486/8870055


def f(x):
    x_points = [0, 1, 2, 3, 4, 5]
    y_points = [12, 14, 22, 39, 58, 77]
    #
    tck = interpolate.splrep(x_points, y_points)
    return interpolate.splev(x, tck)


print(f(1.25))

import numpy as np
import cv2

from numpy import ones, vstack
from numpy.linalg import lstsq


def getStartFrameForFrame(frame):
    return df.query(f'f < {frame} and isBallTrackStart == True').tail(1)['f'].values[0]


def getRadius(x):
    return df.query(f'f <= {frame} and f >= {startFrame}').sort_values('bx').query(f'bx <= {x + 10}').tail(1)['bRadius'].values[0]


# def apply_mask(image, mask, color, alpha=0.5):
#     """Apply the given mask to the image.
#     """
#     for c in range(3):
#         image[:, :, c] = np.where(mask == 1,
#                                   image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
#                                   image[:, :, c])
#     return image

def apply_mask(image, originalImage, mask):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  originalImage[:, :, c],
                                  image[:, :, c])
    return image


###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################

frame = 50
# inputImage = ex4.getMrcnnImgPath("19sec", frame)
inputImage = ex4.getRawImgPath("19sec", frame)
image = cv2.imread(inputImage)
alpha = 0.5
overlay = image.copy()
output = image.copy()

# cv2.rectangle(overlay, (420, 205), (595, 385), (0, 0, 255), -1)
# cv2.rectangle(overlay, (820, 405), (695, 785), (255, 0, 255), -1)

startFrame = getStartFrameForFrame(frame)

spline = splines[startFrame]

r = np.load(ex4.getMrcnnDataPath(videoName, frame)).item()
personMrcnn = ex4.getBestFromMrcnn(ex4.PERSON, r)  # same as biggest i guess??
personMask = personMrcnn['mask']

bx = df.query(f'f == {frame}')['bx'].values[0]
by = df.query(f'f == {frame}')['by'].values[0]
swing = df.query(f'f == {frame}')['swing'].values[0]

splineXMin = min(int(spline["X"][0]), int(spline["X"][len(spline["X"]) - 1]))
splineXMax = max(int(spline["X"][0]), int(spline["X"][len(spline["X"]) - 1]))

minx = min(int(spline["X"][0]), int(bx))
maxx = max(int(spline["X"][0]), int(bx))

minxRadius = getRadius(minx)
maxxRadius = getRadius(maxx)

points = [(minx, minxRadius), (maxx, maxxRadius)]
x_coords, y_coords = zip(*points)
A = vstack([x_coords, ones(len(x_coords))]).T
m, c = lstsq(A, y_coords)[0]
print("Line Solution is y = {m}x + {c}".format(m=m, c=c))


def getRadiusFunction(x):
    return m * x + c

if frame - startFrame < 20:
    for x in range(minx, maxx):
        if x >= splineXMax or x <= splineXMin:
            continue
        tck = spline["tck"]
        y = int(interpolate.splev(x, tck))
        radius = int(getRadiusFunction(x))
        print(x, y, radius)
        cv2.circle(overlay, (x, y), radius, (255, 255, 0))

#
# frame = 190
# # inputImage = ex4.getMrcnnImgPath("19sec", frame)
# inputImage = ex4.getRawImgPath("19sec", frame)
# image = cv2.imread(inputImage)
# alpha = 0.5
# overlay = image.copy()
# output = image.copy()


width = image.shape[1]
height = image.shape[0]
# cv2.rectangle(overlay, (width - 495, height - 40), (width, height - 40), (255, 255, 255), -1)
cv2.putText(overlay, "Stuart Robinson, Durham NC", (width - 480, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
cv2.putText(overlay, "STATS", (470, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (50, 50, 50), 3)

# cv2.rectangle(overlay, (image.shape[1] - 495, 0), (image.shape[1], 40), (255, 255, 255), -1)
# cv2.putText(overlay, "Stuart Robinson, Durham NC", (width - 480, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
cv2.putText(overlay, swing, (30, image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 3)

overlay = apply_mask(overlay, image, personMask)

cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
cv2.imwrite('cvout.png', output)


#TODO - render each raw frame!!!
# remember previous one's overlay.  if current frame doesn't have contrail, use prev frame's contrail and swing word