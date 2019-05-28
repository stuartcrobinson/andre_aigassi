# this is cool http://pythontutor.com/visualize.html#mode=display  interactive visualizer
import copy
import math
import time

import numpy as np
import pandas as pd
from scipy.spatial import distance

import brownlee_maskrcnn.ex4 as ex4

pd.set_option("display.max_rows", 600)

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

df['tbRadiusAndCoords'] = df.apply(lambda x: getTennisBallRadiusAndCoordinates(x, tbBoxes), axis=1)

df[['tbRadius', 'tbx0', 'tby0']] = pd.DataFrame(df['tbRadiusAndCoords'].tolist(), index=df.index)

df = df.drop(errors='ignore',
             columns=['tbx', 'tby', 'rdx1', 'rdx2', 'rdx3', 'rdx4', 'rdx5', 'rdy2', 'rdy4', 'rpx1', 'rpx3', 'rpx5', 'rdy1', 'rdy3', 'rdy5', 'rpy1', 'rpy3', 'rpy5', 'tbRadiusAndCoords', 'racket_distal_delta_1', 'racket_distal_delta_3',
                      'racket_distal_delta_5', 'sideswitched', 'rdΔ1', 'rdΔ2', 'rdΔ3', 'rdΔ4', 'rdΔ5', 'rdΔ1μ6', 'rdΔ1_est', 'tbInR', 'rpx0', 'rpy0', 'rdx0', 'rdy0', 'rdΔ1μ4_est'])
del side6ago

df['tbx1'] = df['tbx0'].shift(1)
df['tbx2'] = df['tbx0'].shift(2)
df['tbx3'] = df['tbx0'].shift(3)
df['tbx4'] = df['tbx0'].shift(4)
df['tby1'] = df['tby0'].shift(1)
df['tby2'] = df['tby0'].shift(2)
df['tby3'] = df['tby0'].shift(3)
df['tby4'] = df['tby0'].shift(4)


# VECTORS
# https://stackoverflow.com/questions/17332759/finding-vectors-with-2-points
# https://stackoverflow.com/a/18514434/8870055 dealing with points and vectors and functions

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            # >>> angle_between((1, 0, 0), (0, 1, 0))
            # 1.5707963267948966
            # >>> angle_between((1, 0, 0), (1, 0, 0))
            # 0.0
            # >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


df['tba0'] = df.apply(lambda r: angle_between(np.array([r.tbx1, r.tby1]), np.array([r.tbx0, r.tby0])), axis=1)
df['tba1'] = df.apply(lambda r: angle_between(np.array([r.tbx2, r.tby2]), np.array([r.tbx0, r.tby0])), axis=1)
df['tba2'] = df.apply(lambda r: angle_between(np.array([r.tbx3, r.tby3]), np.array([r.tbx0, r.tby0])), axis=1)
df['tba3'] = df.apply(lambda r: angle_between(np.array([r.tbx4, r.tby4]), np.array([r.tbx0, r.tby0])), axis=1)

df['tbd0'] = df.apply(lambda r: distLam(r.tbx1, r.tby1, r.tbx0, r.tby0), axis=1)
df['tbd1'] = df.apply(lambda r: distLam(r.tbx2, r.tby2, r.tbx0, r.tby0), axis=1)
df['tbd2'] = df.apply(lambda r: distLam(r.tbx3, r.tby3, r.tbx0, r.tby0), axis=1)
df['tbd3'] = df.apply(lambda r: distLam(r.tbx4, r.tby4, r.tbx0, r.tby0), axis=1)

# now what


# i have all the data??????


# lets make everytyhig fresh
