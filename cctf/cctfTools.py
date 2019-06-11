import os

import cv2
import numpy as np

from cctf.cctfTools import getColorBlock, getDiff, colorizeGrayImg
from cctf.cctfTools import getColorMakingMatrices, getCctf
from satyaStabilization.satyaHomography2018 import alignHomography2018

coordinatesFile = "/Users/stuartrobinson/repos/computervision/andre_aigassi/images/badminton/Badminton_dataset/Badminton_label.csv"
imagesDir = "/Users/stuartrobinson/repos/computervision/andre_aigassi/images/badminton/Badminton_dataset/video_frames"


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


def writeTextTopLeft(image_in, text):
    cv2.putText(img=image_in, text=text, org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[0, 0, 0], lineType=cv2.LINE_AA, thickness=4)
    cv2.putText(img=image_in, text=text, org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[100, 100, 100], lineType=cv2.LINE_AA, thickness=2)


def getImage(frameNumber):
    path = os.path.join(imagesDir, str(frameNumber) + '.jpg')
    # print("path", path)
    image = cv2.imread(path)
    return image


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


def getCctf(g0, g1, g2, g3, g4, g5, g6, c1, c2, c3, c4, c5, c6, black, white, doAlign=False):
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
    s = g0.shape
    # this works for rainbow!
    diff0_1c = colorizeGrayImg(diff0_1, c1)
    diff1_2c = colorizeGrayImg(diff1_2, c2)
    diff2_3c = colorizeGrayImg(diff2_3, c3)
    diff3_4c = colorizeGrayImg(diff3_4, c4)
    diff4_5c = colorizeGrayImg(diff4_5, c5)
    diff5_6c = colorizeGrayImg(diff5_6, c6)
    #
    cctf = diff0_1c + diff1_2c + diff2_3c + diff3_4c + diff4_5c + diff5_6c
    # cctf = cctf *cctf
    cctf = np.minimum(cctf, white)
    cctf = np.maximum(cctf, black)
    cctf = cctf.astype('uint8')
    return cctf


def generateCctfTrackNetBadmintonImagesNpyFile(h):
    # frames = np.load('badmintonProcessedFrames_full_112.npy')
    # for i in range(0, 110):
    #     im = frames[i]
    #     cv2.imshow("asdf", im)
    #     cv2.waitKey(10)
    # quit()
    visAndCoords = getBadmintonCoordinatesAndConf(h)
    w = h
    n_frames = sum(1 for line in open(coordinatesFile)) - 1
    prev = getImage(1)
    prev = cv2.resize(prev, (h, h))
    prev1_gray = prev2_gray = prev3_gray = prev4_gray = prev5_gray = prev6_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    c1, c2, c3, c4, c5, c6, black, white = getColorMakingMatrices(prev1_gray.shape)
    frames = np.zeros([n_frames, h, h, 3], dtype='uint8')
    for i in range(1, n_frames - 1):
        print("frame:", i, " of ", n_frames)
        im = getImage(i + 1)
        frame_color = cv2.resize(im, (w, h))
        frame = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        frame_out = getCctf(frame, prev1_gray, prev2_gray, prev3_gray, prev4_gray, prev5_gray, prev6_gray, c1, c2, c3, c4, c5, c6, black, white, False)
        vis, x, y = visAndCoords[i]
        # tools.writeTextTopLeft(im, str(i) + " of " + str(images.shape[0]))
        frame_out = cv2.resize(frame_out, (w * 4, h * 4))
        cv2.circle(frame_out, (x * 4, y * 4), 10, (0, 0, 0), thickness=2)
        cv2.circle(frame_out, (x * 4, y * 4), 16, (255, 255, 255), thickness=2)
        cv2.imshow("asdf2", frame_out)
        cv2.waitKey(100)
        # frames[i] = frame_out
        # print(frames)
        # out.write(frame_out)  # https://stackoverflow.com/a/50076149/8870055
        prev6_gray = prev5_gray  # .copy()
        prev5_gray = prev4_gray  # .copy()
        prev4_gray = prev3_gray  # .copy()
        prev3_gray = prev2_gray  # .copy()
        prev2_gray = prev1_gray  # .copy()
        prev1_gray = frame  # .copy()
    # np.save('badmintonProcessedFrames_full_' + str(h) + '.npy', frames)
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


def runQaOnTracknetTrainingData(targetHeight):
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
        im = cv2.resize(im, (targetHeight * 3, targetHeight * 3))

        writeTextTopLeft(im, str(frameNumber) + " of " + str(numFrames))
        cv2.circle(im, (x * 3, y * 3), 10, (0, 255, 0), thickness=2)
        cv2.imshow("asdf2", im)
        cv2.waitKey(100)


def runQaOnNpyCctfFrames(imgHeight):
    images = np.load(f'/Users/stuartrobinson/repos/computervision/andre_aigassi/cctf/badmintonProcessedFrames_full_{imgHeight}_safe.npy')
    visAndCoords = getBadmintonCoordinatesAndConf(imgHeight)

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
