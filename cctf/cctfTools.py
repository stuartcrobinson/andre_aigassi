import os

import cv2
import numpy as np

import imgTools as tools
from satyaStabilization.satyaHomography2018 import alignHomography2018

trackNetBadmintonTDCoordinatesFile = "/Users/stuartrobinson/repos/computervision/andre_aigassi/images/badminton/Badminton_dataset/Badminton_label.csv"
trackNetBadmintonTDImagesDir = "/Users/stuartrobinson/repos/computervision/andre_aigassi/images/badminton/Badminton_dataset/video_frames"


def getImage(frameNumber):
    path = os.path.join(trackNetBadmintonTDImagesDir, str(frameNumber) + '.jpg')
    image = cv2.imread(path)
    return image


def getBwImage(frameNumber, resizeHeight=-1, doSquare=False):
    path = os.path.join(trackNetBadmintonTDImagesDir, str(frameNumber) + '.jpg')
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if resizeHeight > 0:
        image = resizeImage(image, resizeHeight)
    if doSquare:
        image = centerSquareCrop(image)
    return image


def getNumTracknetTDFrames():
    numFrames = sum(1 for line in open(trackNetBadmintonTDCoordinatesFile)) - 1
    return numFrames
    # return 100


def getStartFrame():
    return 75


def getColorBlock(shape, r, g, b):
    colors = np.zeros((shape[0], shape[1], 3), np.int)
    colors[:] = (b, g, r)
    return colors


def colorizeGrayImg(imGray, colors):
    '''accepts only bw image'''
    imGrayRgb = cv2.cvtColor(imGray, cv2.COLOR_GRAY2BGR)
    fpct = imGrayRgb / 255
    fc = colors * fpct
    return fc


def getDiff(img1, img2):
    '''img1 - img2'''
    '''should probably use opencv diff instead'''
    stack = np.dstack([img1, img2])
    diff = np.absolute(np.dot(stack, [1, -1])).astype('uint8')
    return diff


def getResizeFactor(targetHeight, im):
    height = im.shape[0]
    resizeFactr = height / targetHeight
    return resizeFactr


def getSquareCropLeftRight(im):
    # crop image to center square
    width = im.shape[1]
    height = im.shape[0]
    centerColumn = width // 2
    halfHeight = height // 2
    left = centerColumn - halfHeight
    right = centerColumn + halfHeight
    return left, right


def getBadmintonVisAndCoords_resizedAndSquared(h):
    sampleImage = getBwImage(1)
    resizeFactor = getResizeFactor(h, sampleImage)
    left, right = getSquareCropLeftRight(sampleImage)

    n_rows = sum(1 for line in open(trackNetBadmintonTDCoordinatesFile)) - 1
    visAndCoordinates = np.zeros([n_rows, 3], dtype=int)
    #
    with open(trackNetBadmintonTDCoordinatesFile) as f:
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


def getColorMakingMatrices(shape):
    '''when applied to pairwise image diffs, and then when they're added together, these results provide a rainbow-spectrum image'''
    s = shape
    c1 = (getColorBlock(s, 255, 0, 0))
    c2 = (getColorBlock(s, 0, 126, 0))
    c3 = (getColorBlock(s, 255, 126, 0))
    c4 = (getColorBlock(s, -255, 126, 0))
    c5 = (getColorBlock(s, -75, -75, 255))
    c6 = (getColorBlock(s, 255, 0, 255))
    black = getColorBlock(s, 0, 0, 0)
    white = getColorBlock(s, 255, 255, 255)
    return c1, c2, c3, c4, c5, c6, black, white


def getCctf(g0, g1, g2, g3, g4, g5, g6, c1, c2, c3, c4, c5, c6, black, white, doAlign=False, brightness=1):
    ''' g for gray image, c for color-kinda matrix.  not actual colors, some neg values'''
    if doAlign:
        g0 = cv2.cvtColor(alignHomography2018(cv2.cvtColor(g0, cv2.COLOR_GRAY2BGR), g0, g3), cv2.COLOR_BGR2GRAY).copy()
        g1 = cv2.cvtColor(alignHomography2018(cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR), g1, g3), cv2.COLOR_BGR2GRAY).copy()
        g2 = cv2.cvtColor(alignHomography2018(cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR), g2, g3), cv2.COLOR_BGR2GRAY).copy()
        g3 = cv2.cvtColor(alignHomography2018(cv2.cvtColor(g3, cv2.COLOR_GRAY2BGR), g3, g3), cv2.COLOR_BGR2GRAY).copy()
        g4 = cv2.cvtColor(alignHomography2018(cv2.cvtColor(g4, cv2.COLOR_GRAY2BGR), g4, g3), cv2.COLOR_BGR2GRAY).copy()
        g5 = cv2.cvtColor(alignHomography2018(cv2.cvtColor(g5, cv2.COLOR_GRAY2BGR), g5, g3), cv2.COLOR_BGR2GRAY).copy()
        g6 = cv2.cvtColor(alignHomography2018(cv2.cvtColor(g6, cv2.COLOR_GRAY2BGR), g6, g3), cv2.COLOR_BGR2GRAY).copy()
    diff0_1 = getDiff(g0, g1)
    diff1_2 = getDiff(g1, g2)
    diff2_3 = getDiff(g2, g3)
    diff3_4 = getDiff(g3, g4)
    diff4_5 = getDiff(g4, g5)
    diff5_6 = getDiff(g5, g6)
    #
    diff0_1c = colorizeGrayImg(diff0_1, c1)
    diff1_2c = colorizeGrayImg(diff1_2, c2)
    diff2_3c = colorizeGrayImg(diff2_3, c3)
    diff3_4c = colorizeGrayImg(diff3_4, c4)
    diff4_5c = colorizeGrayImg(diff4_5, c5)
    diff5_6c = colorizeGrayImg(diff5_6, c6)
    #
    cctf = diff0_1c + diff1_2c + diff2_3c + diff3_4c + diff4_5c + diff5_6c
    if brightness != 1:
        cctf = cctf * brightness
    cctf = np.minimum(cctf, white)
    cctf = np.maximum(cctf, black)
    cctf = cctf.astype('uint8')
    return cctf


def generateCctfTrackNetBadmintonImagesNpyFile(targetHeight=-1, startFrame=1, endFrame=getNumTracknetTDFrames(), brightness=1):
    sampleImage = getBwImage(1)
    defaultHeight = sampleImage.shape[0]
    print("default im shape:", sampleImage.shape)
    if targetHeight < 0:
        targetHeight = defaultHeight

    print("targetHeight:", targetHeight)

    # frames = np.load('badmintonProcessedFrames_full_112.npy')
    # for i in range(0, 110):
    #     im = frames[i]
    #     cv2.imshow("asdf", im)
    #     cv2.waitKey(10)
    # quit()
    visAndCoords = getBadmintonVisAndCoords_resizedAndSquared(targetHeight)
    n_frames = getNumTracknetTDFrames()
    prev1 = prev2 = prev3 = prev4 = prev5 = prev6 = getBwImage(1, targetHeight, doSquare=True)
    c1, c2, c3, c4, c5, c6, black, white = getColorMakingMatrices(prev1.shape)
    frames = np.zeros([n_frames, targetHeight, targetHeight, 3], dtype='uint8')
    for i in range(n_frames - 1):
        frameNumber = i + 1
        if frameNumber < startFrame:
            continue
        if frameNumber > endFrame:
            break
        print("frame:", frameNumber, " of ", n_frames)
        curr = getBwImage(i + 1, targetHeight, doSquare=True)
        cctfIm = getCctf(curr, prev1, prev2, prev3, prev4, prev5, prev6, c1, c2, c3, c4, c5, c6, black, white, False, brightness)
        cctfIm = labelFrameNumber(cctfIm, frameNumber, n_frames);
        cctfIm = encircleBirdie(cctfIm, frameNumber, visAndCoords)
        cv2.imshow("asdf2", cctfIm)
        cv2.waitKey(100)
        frames[i] = cctfIm
        prev6 = prev5
        prev5 = prev4
        prev4 = prev3
        prev3 = prev2
        prev2 = prev1
        prev1 = curr
    np.save('badmintonProcessedFrames_full_' + str(targetHeight) + '.npy', frames)
    pass


def generateCctfVideoFileFromVideo():
    # Read input video
    # cap = cv2.VideoCapture('video.mp4')
    # cap = cv2.VideoCapture('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/forehand.mp4')
    # cap = cv2.VideoCapture('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/badminton_video/raw/Longest rally in badminton history (Men´s singles).mp4')

    # cap = cv2.VideoCapture('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/19sec.mov')
    cap = cv2.VideoCapture('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/badminton/raw/Longest rally in badminton history (Men´s singles).mp4')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) / 1.5)
    h = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) / 1.5)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    print(n_frames, w, h, fps, fourcc)

    # out = cv2.VideoWriter('output/cctf_video_out.avi', fourcc, fps, (w, 2*h))
    out = cv2.VideoWriter('/Users/stuartrobinson/repos/computervision/andre_aigassi/output/cctf_video_out.avi', fourcc, fps, (w, h))

    _, prev = cap.read()
    prev = cv2.resize(prev, (w, h))
    prev1_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev2_gray = prev1_gray.copy()
    prev3_gray = prev1_gray.copy()
    prev4_gray = prev1_gray.copy()
    prev5_gray = prev1_gray.copy()
    prev6_gray = prev1_gray.copy()
    c1, c2, c3, c4, c5, c6, black, white = getColorMakingMatrices(prev1_gray.shape)
    for i in range(n_frames - 2):
        success, frame_color = cap.read()
        print("frame:", i)
        if not success:
            break
        frame_color = cv2.resize(frame_color, (w, h))
        frame = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        #
        # diff = getDiff(frame, prev1_gray)
        # diff_grayRgb = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        #
        if i > 98:
            # cctfAligned = getCctf(frame, prev1_gray, prev2_gray, prev3_gray, prev4_gray, prev5_gray, prev6_gray, True)
            cctfUnaligned = getCctf(frame, prev1_gray, prev2_gray, prev3_gray, prev4_gray, prev5_gray, prev6_gray, c1, c2, c3, c4, c5, c6, black, white, False)
            #
            # writeTextTopLeft(cctfAligned, 'aligned')
            # writeTextTopLeft(cctfUnaligned, 'unaligned')
            # frame_out = cv2.hconcat([frame, prev1_gray, prev2_gray, prev3_gray])
            # frame_out = cv2.hconcat([frame, prev1_gray, prev2_gray, prev3_gray])
            # frame_out = cctf
            frame_out = cctfUnaligned
            # frame_out = cv2.vconcat([cctfUnaligned, frame_color])
            #
            cv2.imshow("asdf", frame_out)
            cv2.waitKey(10)
            out.write(frame_out)  # https://stackoverflow.com/a/50076149/8870055
        prev6_gray = prev5_gray  # .copy()
        prev5_gray = prev4_gray  # .copy()
        prev4_gray = prev3_gray  # .copy()
        prev3_gray = prev2_gray  # .copy()
        prev2_gray = prev1_gray  # .copy()
        prev1_gray = frame  # .copy()
    #
    # Release video
    cap.release()
    out.release()
    # Close windows
    cv2.destroyAllWindows()


def resizeImage(im, targetHeight):
    resizeFactor = im.shape[0] / targetHeight
    targetWidth = int(im.shape[1] / resizeFactor)
    return cv2.resize(im, (targetWidth, targetHeight))


def centerSquareCrop(im):
    h = im.shape[0]
    w = im.shape[1]
    centerColumn = w // 2
    halfHeight = h // 2
    left = centerColumn - halfHeight
    right = centerColumn + halfHeight
    return im[:, left:right]


def encircleBirdie(im, frameNumber, visAndCoords):
    visibility, x, y = visAndCoords[frameNumber - 1]
    cv2.circle(im, (x, y), 13, (0, 255, 0), thickness=2)
    cv2.circle(im, (x, y), 15, (255, 255, 255), thickness=2)
    return im


def labelFrameNumber(im, frameNumber, numFrames):
    tools.writeTextTopLeft(im, str(frameNumber) + " of " + str(numFrames))
    return im

from numpy import linalg as LA

def runQaOnTracknetTrainingDataPairsOnly(targetHeight=-1, startFrame=1, endFrame=getNumTracknetTDFrames(), brightness=1):
    sampleImage = getBwImage(1)
    defaultHeight = sampleImage.shape[0]
    print("default im shape:", sampleImage.shape)
    if targetHeight < 0:
        targetHeight = defaultHeight

    print("targetHeight:", targetHeight)
    visAndCoords = getBadmintonVisAndCoords_resizedAndSquared(targetHeight)
    n_frames = getNumTracknetTDFrames()
    prev1 = getBwImage(1, targetHeight, doSquare=True)
    for i in range(n_frames - 1):
        frameNumber = i + 1
        if frameNumber < startFrame:
            continue
        if frameNumber > endFrame:
            break
        # print("frame:", frameNumber, " of ", n_frames)
        curr = getBwImage(i + 1, targetHeight, doSquare=True)
        diff = getDiff(curr, prev1)
        black = getColorBlock(diff.shape, 0, 0, 0)
        if diff == black:
            print("is black!!!!!!!!!!!")
        normCurr = LA.norm(curr)        #these are not helpful for identifying repeated frames
        normPrev= LA.norm(prev1)
        normDiff = normCurr - normPrev
        maxElement = np.amax(diff)
        theMean = diff.mean()
        theSum = diff.sum()
        print("frame: {}  max: {:4s}  mean: {:.3f} sum: {:7} normdiff: {:.3f}".format(frameNumber, str(maxElement), theMean, theSum, abs(normDiff)))
        diff = labelFrameNumber(diff, frameNumber, n_frames);
        diff = encircleBirdie(diff, frameNumber, visAndCoords)
        # print("diff.shape", diff.shape)
        cv2.imshow("asdf2", diff)
        cv2.waitKey(1000)
        prev1 = curr
    pass

def recordBadFrames(targetHeight=-1, startFrame=1, endFrame=getNumTracknetTDFrames()):
    sampleImage = getBwImage(1)
    defaultHeight = sampleImage.shape[0]
    print("default im shape:", sampleImage.shape)
    if targetHeight < 0:
        targetHeight = defaultHeight

    print("targetHeight:", targetHeight)
    visAndCoords = getBadmintonVisAndCoords_resizedAndSquared(targetHeight)
    n_frames = getNumTracknetTDFrames()
    prev1 = getBwImage(1, targetHeight, doSquare=True)
    for i in range(n_frames - 1):
        frameNumber = i + 1
        if frameNumber < startFrame:
            continue
        if frameNumber > endFrame:
            break
        # print("frame:", frameNumber, " of ", n_frames)
        curr = getBwImage(i + 1, targetHeight, doSquare=True)
        diff = getDiff(curr, prev1)
        black = getColorBlock(diff.shape, 0, 0, 0)
        if diff == black:
            print("is black!!!!!!!!!!!")
        normCurr = LA.norm(curr)        #these are not helpful for identifying repeated frames
        normPrev= LA.norm(prev1)
        normDiff = normCurr - normPrev
        maxElement = np.amax(diff)
        theMean = diff.mean()
        theSum = diff.sum()
        print("frame: {}  max: {:4s}  mean: {:.3f} sum: {:7} normdiff: {:.3f}".format(frameNumber, str(maxElement), theMean, theSum, abs(normDiff)))
        diff = labelFrameNumber(diff, frameNumber, n_frames);
        diff = encircleBirdie(diff, frameNumber, visAndCoords)
        # print("diff.shape", diff.shape)
        cv2.imshow("asdf2", diff)
        cv2.waitKey(100)
        prev1 = curr
    pass


def runQaOnTracknetTrainingData(targetHeight=-1, startFrame=1, endFrame=getNumTracknetTDFrames()):
    """
    doing QA on I-No Liao's TrackNet's badminton tracking traing data from:
    https://inoliao.github.io/CoachAI/
    https://drive.google.com/uc?export=download&id=1ZgoGm5y3_fSwzWLBFe_4Zu4LnMMkUd0J

    this file displays each image in the training data with a green circle around the coordinates listed in Badminton_label.csv

    purpose is to make sure the listed coordinates are actually effectively tracking the birdie.

    also to make sure we're cropping and resizing correctly. ✓✓

    result -- data looks great !!! thank you I-No Liao!!!!
    """

    sampleImage = getBwImage(1)
    defaultHeight = sampleImage.shape[0]
    print("default im shape:", sampleImage.shape)
    if targetHeight < 0:
        targetHeight = defaultHeight

    numFrames = getNumTracknetTDFrames()

    visAndCoords = getBadmintonVisAndCoords_resizedAndSquared(targetHeight)

    print("visAndCoords.shape:", visAndCoords.shape)
    for i in range(numFrames):
        frameNumber = i + 1
        if frameNumber < startFrame:
            continue
        if frameNumber > endFrame:
            break
        im = getBwImage(frameNumber)
        im = centerSquareCrop(im)
        maxElement = np.amax(im)
        theMean = im.mean()
        theSum = im.sum()
        print("frame: {}  max: {:7s}   mean: {:.3f}  sum: {}".format(frameNumber, str(maxElement), theMean, theSum))
        if targetHeight != defaultHeight:
            im = resizeImage(im, targetHeight)

        tools.writeTextTopLeft(im, str(frameNumber) + " of " + str(numFrames))
        im = encircleBirdie(im, frameNumber, visAndCoords)
        cv2.imshow("asdf2", im)
        cv2.waitKey(300)


def runQaOnNpyCctfFrames(imgHeight):
    images = np.load(f'/Users/stuartrobinson/repos/computervision/andre_aigassi/cctf/badmintonProcessedFrames_full_{imgHeight}_safe.npy')
    visAndCoords = getBadmintonVisAndCoords_resizedAndSquared(imgHeight)

    black = getColorBlock(images[0].shape, 0, 0, 0)
    white = getColorBlock(images[0].shape, 255, 255, 255)
    for i in range(images.shape[0]):
        im = images[i]
        im = im * 10
        im = np.minimum(im, white)
        im = np.maximum(im, black)
        im = im.astype('uint8')
        im = cv2.resize(im, (imgHeight * 3, imgHeight * 3))
        vis, x, y = visAndCoords[i]
        # tools.writeTextTopLeft(im, str(i) + " of " + str(images.shape[0]))
        cv2.circle(im, (x * 3, y * 3), 10, (0, 255, 0), thickness=2)
        cv2.imshow("asdf2", im)
        cv2.waitKey(100)
