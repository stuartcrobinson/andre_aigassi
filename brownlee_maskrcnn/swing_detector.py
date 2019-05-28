# how?

# look for a series of 5 frames where the right wrist goes cumulative D distance

# D = 250 pixels?  make it relative to back height?

'''

open all mrcnn npy files to get 'sports ball' bounding boxes

clean:  if ball doens't move by more than 25 pixels AND it's not in line with prev two balls, AND it's 100 pixels or more away from prev
            throw it out


0.  generate list of ball velocities (2d) per frame
        v = prev frame to current frame
                if not in prev frame, use frame before that (but no more?)
        v = slope of the line of the ball movement from prev frame




-------

so, for each pose data file
... no just open all the pose data files

store files as list of biggest persons

make 2 lists:
right wrist coordinates
neck coordinates


per index in arrays,

    get 5th index in the future, k
    if wrist changes body sides between i and k, check wrist speed - cumulative between positions
                    if wrist position is ever 0, take mean of surrounding coordinates
    if cumulative wrist speed > cutoff
        identify swing!  check backhand/forehand
    find frame where tennis ball is opposite the racket from camera?
    find frame where tennis ball direction changes?

'''

import copy

import numpy as np
from scipy.spatial import distance

import brownlee_maskrcnn.ex4 as ex4

print(distance.euclidean((1, 2), (1, 2)))
wristCoords = []
neckCoords = []


# stanford_single_swing_000000000000_keypoints.json

# frameNumber starts at 0
#
#
# # poseFile = '/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/pose/data/19sec/000099_keypoints.json'
# poseFile = '/Users/stuartrobinson/repos/computervision/andre_aigassi/images/img/out/000099_keypoints.json'
# poses = ex4.readJsonFile(poseFile)
# people = poses['people']
#
# for person in people:
#     size = ex4.getPosePersonSize(person)
#     print("size", size)
#     print(json.dumps(ex4.getReadablePose(person), indent=4))
# print('######################################################################')
# personPose = ex4.getMainPlayerPose(poses)
# print(json.dumps(ex4.getReadablePose(personPose), indent=4))


def getPlayerPoses(videoName):
    numFrames = len(ex4.getRawVideoFrames(videoName))
    playerPoses = []
    for frameNumber in range(1, numFrames + 1):
        posesPath = ex4.getPosesPath(videoName, frameNumber)
        poses = ex4.readJsonFile(posesPath)
        personPose = ex4.getMainPlayerPose(poses)
        playerPoses += [personPose]
    return playerPoses


def getPlayerBoxes(videoName):
    numFrames = len(ex4.getRawVideoFrames(videoName))
    playerBoxes = [{} for i in range(numFrames)]
    for i in range(numFrames):
        try:
            frameNumber = i + 1
            r = np.load(ex4.getMrcnnDataPath(videoName, frameNumber)).item()
            personMrcnn = ex4.getBiggestMrcnnPerson(r)
            playerBox = personMrcnn['roi']
            playerBoxes[i] = playerBox
        except:
            continue
    return playerBoxes


def getRacketBoxes(videoName):
    # racketMrcnn = getBestFromMrcnn(TENNIS_RACKET, r)
    # racketBox = racketMrcnn['roi']
    numFrames = len(ex4.getRawVideoFrames(videoName))
    racketBoxes = [{} for i in range(numFrames)]
    for i in range(numFrames):
        try:
            frameNumber = i + 1
            r = np.load(ex4.getMrcnnDataPath(videoName, frameNumber)).item()
            racketMrcnn = ex4.getBestFromMrcnn(ex4.TENNIS_RACKET, r)
            racketBox = racketMrcnn['roi'] if 'roi' in racketMrcnn else None
            racketBoxes[i] = racketBox
        except:
            continue
    return racketBoxes


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


videoName = "19sec"

numFrames = len(ex4.getRawVideoFrames(videoName))

playerPoses = getPlayerPoses(videoName)
playerBoxes = getPlayerBoxes(videoName)
racketBoxes = getRacketBoxes(videoName)
# racketMasks = getRacketMasks(videoName)

analysis = ex4.readJsonFile(ex4.getAnalysisPath(videoName))

if len(analysis) < numFrames:
    analysis = [{} for i in range(numFrames)]

for i in range(numFrames):
    print(i)
    pose = playerPoses[i]
    personBox = playerBoxes[i]
    racketBox = racketBoxes[i]
    side = ex4.getActionSide(personBox, pose, racketBox)
    analysis[i]['racket_side'] = side

# racketMasks = getRacketMasks(videoName)

for i in range(numFrames):
    print(i)
    frameNumber = i + 1
    pose = playerPoses[i]
    r = np.load(ex4.getMrcnnDataPath(videoName, frameNumber)).item()
    racketMrcnn = ex4.getBestFromMrcnn(ex4.TENNIS_RACKET, r)
    racketMask = racketMrcnn['mask'] if 'mask' in racketMrcnn else None
    proximalCoords, distalCoords = getRacketExtremeCoords(racketMask,
                                                          ex4.getBodyPartCoordinates(ex4.rightWristNumber, pose),
                                                          ex4.getBodyPartCoordinates(ex4.rightElbowNumber, pose),
                                                          ex4.getBodyPartCoordinates(ex4.rightShoulderNumber, pose))
    analysis[i]['racket_distal'] = distalCoords
    analysis[i]['racket_proximal'] = proximalCoords

for i in range(5, numFrames):
    if 'racket_distal' not in analysis[i]:
        continue
    print(i, analysis[i]['racket_distal'])
    print(i, analysis[i - 1]['racket_distal'])
    print(i, analysis[i - 3]['racket_distal'])
    print(i, analysis[i - 5]['racket_distal'])
    if analysis[i]['racket_distal'] is not None:
        if analysis[i - 1]['racket_distal'] is not None:
            analysis[i]['racket_distal_delta_1'] = distance.euclidean(analysis[i]['racket_distal'], analysis[i - 1]['racket_distal'])
        if analysis[i - 3]['racket_distal'] is not None:
            analysis[i]['racket_distal_delta_3'] = distance.euclidean(analysis[i]['racket_distal'], analysis[i - 3]['racket_distal'])
        if analysis[i - 5]['racket_distal'] is not None:
            analysis[i]['racket_distal_delta_5'] = distance.euclidean(analysis[i]['racket_distal'], analysis[i - 5]['racket_distal'])
        deltas = [analysis[i - 0]['racket_distal_delta_1'],
                  analysis[i - 1]['racket_distal_delta_1'],
                  analysis[i - 2]['racket_distal_delta_1'],
                  analysis[i - 3]['racket_distal_delta_1'],
                  analysis[i - 4]['racket_distal_delta_1']]
        deltas = list(filter(None.__ne__, deltas))  # https://stackoverflow.com/a/54260099/8870055

        TODO start here

        working on getting measurements for distal racket tip speed to determine when a swing is happening.

            getting ave speed over 5-frame span.  determine threshold for swings.  count how many actual swings there in are 19sec

        next, get tennis ball coordinates, and add tennis ball velocities to 'analysis'
        then add "turn" key for when the velocity changes abruptly
        wait shit ... how to encode velocity?  since slope is bidirectional
        maybe encode velocity as a vector?
    to find turn, calculate the angle between the vectors

    when theres a turn and there's a fast racket motion and there's a racketsidechange, there should be a swing


        analysis[i]['racket_distal_mean_delta_5'] = np.mean(deltas)

for i in range(numFrames):
    print(i, analysis[i])

    if side == None:
        continue

#
# for fileName in inputFiles:
#     frameNumber = int(fileName.split('stanford_single_swing_')[1].split('_keypoints')[0])
#     imgPath = join(posesDir, fileName)
#     print("processing: ", frameNumber, imgPath)
#     #
#     # TODO start here -- analyze pose data to detect swing type
#     midHipCoords = ex4.getBodyPartCoordinates(ex4.midHipPartNumber, {})
#     print(neckCoordinates)
#
# numFrames = len(ex4.getRawVideoFrames(videoName))
#
# start = time.time()
#
# for frameNumber in range(1, numFrames + 1):
#     inputImgPath = ex4.getRawImgPath(videoName, frameNumber)
#     posesPath = getPosesPath(videoName, frameNumber)
#     outputImgPath = getSegmentedImgOutputPath(videoName, frameNumber)
#     outputDataPath = getSegmentedDataOutputPath(videoName, frameNumber)
#     # if frameNumber < 98:
