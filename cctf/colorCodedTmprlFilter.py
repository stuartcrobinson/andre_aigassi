import cv2
import numpy as np

from cctf.cctfTools import getColor
from cctf.qaTrackNetBadmintonData import coordinatesFile, getImage, getBadmintonCoordinatesAndConf
from satyaStabilization.satyaHomography2018 import alignHomography2018


# im0 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/badminton/Badminton_dataset/video_frames/4.jpg')
# im1 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/badminton/Badminton_dataset/video_frames/5.jpg')
# im2 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/badminton/Badminton_dataset/video_frames/6.jpg')
# im3 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/badminton/Badminton_dataset/video_frames/7.jpg')
# im4 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/badminton/Badminton_dataset/video_frames/8.jpg')
# im5 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/badminton/Badminton_dataset/video_frames/9.jpg')

#
# im0 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/forehand/000006.png')
# im1 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/forehand/000007.png')
# im2 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/forehand/000008.png')
# im3 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/forehand/000009.png')
# im4 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/forehand/000010.png')
# im5 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/forehand/000011.png')
#
# im0 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/19sec/000016.png')
# im1 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/19sec/000017.png')
# im2 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/19sec/000018.png')
# im3 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/19sec/000019.png')
# im4 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/19sec/000020.png')
# im5 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/19sec/000021.png')
# im6 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/19sec/000022.png')
#
# resizeFactor = 1
#
# im0 = cv2.bye(im0, (im0.shape[1] // resizeFactor, im0.shape[0] // resizeFactor))
# im1 = cv2.resize(im1, (im1.shape[1] // resizeFactor, im1.shape[0] // resizeFactor))
# im2 = cv2.resize(im2, (im2.shape[1] // resizeFactor, im2.shape[0] // resizeFactor))
# im3 = cv2.resize(im3, (im3.shape[1] // resizeFactor, im3.shape[0] // resizeFactor))
# im4 = cv2.resize(im4, (im4.shape[1] // resizeFactor, im4.shape[0] // resizeFactor))
# im5 = cv2.resize(im5, (im5.shape[1] // resizeFactor, im5.shape[0] // resizeFactor))
# im6 = cv2.resize(im6, (im5.shape[1] // resizeFactor, im5.shape[0] // resizeFactor))


def displayAsColor(im, colors):
    imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imGrayRgb = cv2.cvtColor(imGray, cv2.COLOR_GRAY2BGR)
    fpct = imGrayRgb / 255
    fc = colors * fpct
    fc = fc.astype('uint8')
    cv2.imshow("asdf", fc)
    cv2.waitKey(200)
    return fc, imGray


def changeColor(imGray, colors):
    '''accepts only bw image'''
    imGrayRgb = cv2.cvtColor(imGray, cv2.COLOR_GRAY2BGR)
    fpct = imGrayRgb / 255
    fc = colors * fpct
    return fc


def getDiff(img1, img2):
    '''img1 - img2'''
    stack = np.dstack([img1, img2])
    diff = np.absolute(np.dot(stack, [1, -1])).astype('uint8')
    return diff


def writeTextTopLeft(image_in, text):
    cv2.putText(img=image_in, text=text, org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[0, 0, 0], lineType=cv2.LINE_AA, thickness=4)
    cv2.putText(img=image_in, text=text, org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[100, 100, 100], lineType=cv2.LINE_AA, thickness=2)


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
    diff0_1c = changeColor(diff0_1, c1)
    diff1_2c = changeColor(diff1_2, c2)
    diff2_3c = changeColor(diff2_3, c3)
    diff3_4c = changeColor(diff3_4, c4)
    diff4_5c = changeColor(diff4_5, c5)
    diff5_6c = changeColor(diff5_6, c6)
    #
    cctf = diff0_1c + diff1_2c + diff2_3c + diff3_4c + diff4_5c + diff5_6c
    # cctf = cctf *cctf
    cctf = np.minimum(cctf, white)
    cctf = np.maximum(cctf, black)
    cctf = cctf.astype('uint8')
    return cctf


def getColorMakingMatrices(shape):
    s = shape
    c1 = (getColor(s, 255, 0, 0))
    c2 = (getColor(s, 0, 126, 0))
    c3 = (getColor(s, 255, 126, 0))
    c4 = (getColor(s, -255, 126, 0))
    c5 = (getColor(s, -75, -75, 255))
    c6 = (getColor(s, 255, 0, 255))
    black = getColor(s, 0, 0, 0)
    white = getColor(s, 255, 255, 255)
    return c1, c2, c3, c4, c5, c6, black, white


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
        # if i > 100:
        #     break
        im = getImage(i + 1)
        frame_color = cv2.resize(im, (w, h))
        frame = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        frame_out = getCctf(frame, prev1_gray, prev2_gray, prev3_gray, prev4_gray, prev5_gray, prev6_gray, c1, c2, c3, c4, c5, c6, black, white, False)
        vis, x, y = visAndCoords[i]
        # tools.writeTextTopLeft(im, str(i) + " of " + str(images.shape[0]))
        frame_out = cv2.resize(frame_out, (w*4, h*4))
        cv2.circle(frame_out, (x*4, y*4), 10, (0, 0, 0), thickness=2)
        cv2.circle(frame_out, (x*4, y*4), 16, (255, 255, 255), thickness=2)
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

# generateCctfVideoFileFromVideo()
generateCctfTrackNetBadmintonImagesNpyFile(224)
# generateCctfTrackNetBadmintonImagesNpyFile(112)

#TODO start here.  by running this file.  a lot of things are messed up while debugging.
# trying to check if coordinates are correct over cctf images.
#rebuild X and Y.  use frame number as indices.  not frameNumber-1.
# visually check coordinates at each step.
# maybe getting screwed up during image resize for some reason????