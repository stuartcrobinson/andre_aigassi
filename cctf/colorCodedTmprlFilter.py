import cv2
import numpy as np

from satyaHomography2018 import alignHomography2018

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

im0 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/19sec/000016.png')
im1 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/19sec/000017.png')
im2 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/19sec/000018.png')
im3 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/19sec/000019.png')
im4 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/19sec/000020.png')
im5 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/19sec/000021.png')
im6 = cv2.imread('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/frames/raw/19sec/000022.png')

resizeFactor = 1

im0 = cv2.bye(im0, (im0.shape[1] // resizeFactor, im0.shape[0] // resizeFactor))
im1 = cv2.resize(im1, (im1.shape[1] // resizeFactor, im1.shape[0] // resizeFactor))
im2 = cv2.resize(im2, (im2.shape[1] // resizeFactor, im2.shape[0] // resizeFactor))
im3 = cv2.resize(im3, (im3.shape[1] // resizeFactor, im3.shape[0] // resizeFactor))
im4 = cv2.resize(im4, (im4.shape[1] // resizeFactor, im4.shape[0] // resizeFactor))
im5 = cv2.resize(im5, (im5.shape[1] // resizeFactor, im5.shape[0] // resizeFactor))
im6 = cv2.resize(im6, (im5.shape[1] // resizeFactor, im5.shape[0] // resizeFactor))


def getColor(shape, r, g, b):
    colors = np.zeros((shape[0], shape[1], 3), np.int)
    colors[:] = (b, g, r)
    return colors


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
    # fc = fc*4         #no. it doesn't work to brighten per frame cos "colors" is meaningless as color
    # fc = np.minimum(fc, colors)
    # fc = np.maximum(fc, getColor(fc.shape, 0, 0, 0))
    # fc = fc.astype('uint8')
    # cv2.imshow("asdf", fc)
    # cv2.waitKey(50)
    return fc


def getDiff(img1, img2):
    '''img1 - img2'''
    stack = np.dstack([img1, img2])
    diff = np.absolute(np.dot(stack, [1, -1])).astype('uint8')
    return diff


fGray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
fGray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
fGray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
fGray3 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
fGray4 = cv2.cvtColor(im4, cv2.COLOR_BGR2GRAY)
fGray5 = cv2.cvtColor(im5, cv2.COLOR_BGR2GRAY)
fGray6 = cv2.cvtColor(im6, cv2.COLOR_BGR2GRAY)

diff0_1 = getDiff(fGray0, fGray1)
diff1_2 = getDiff(fGray1, fGray2)
diff2_3 = getDiff(fGray2, fGray3)
diff3_4 = getDiff(fGray3, fGray4)
diff4_5 = getDiff(fGray4, fGray5)
diff5_6 = getDiff(fGray5, fGray6)

# # this works for rainbow!
# diff0_1c = changeColor(diff0_1, getColor(im0.shape, 255, 0, 0))
# diff1_2c = changeColor(diff1_2, getColor(im0.shape, 0, 126, 0))
# diff2_3c = changeColor(diff2_3, getColor(im0.shape, 255, 126, 0))
# diff3_4c = changeColor(diff3_4, getColor(im0.shape, -255, 126, 0))
# diff4_5c = changeColor(diff4_5, getColor(im0.shape, 255, 0, 255))

# this works for rainbow!
diff0_1c = changeColor(diff0_1, getColor(im0.shape, 255, 0, 0))
diff1_2c = changeColor(diff1_2, getColor(im0.shape, 0, 126, 0))
diff2_3c = changeColor(diff2_3, getColor(im0.shape, 255, 126, 0))
diff3_4c = changeColor(diff3_4, getColor(im0.shape, -255, 126, 0))
diff4_5c = changeColor(diff4_5, getColor(im0.shape, -75, -75, 255))
diff5_6c = changeColor(diff5_6, getColor(im0.shape, 255, 0, 255))

cctf = diff0_1c + diff1_2c + diff2_3c + diff3_4c + diff4_5c + diff5_6c
cctf = cctf * 1.5
cctf = np.minimum(cctf, getColor(im0.shape, 255, 255, 255))
cctf = np.maximum(cctf, getColor(im0.shape, 0, 0, 0))
cctf = cctf.astype('uint8')

cv2.imshow("asdf", cctf)
cv2.waitKey(1000)
cv2.waitKey(1000)

def writeTextTopLeft(image_in, text):
    cv2.putText(img=image_in, text=text, org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[0, 0, 0], lineType=cv2.LINE_AA, thickness=4)
    cv2.putText(img=image_in, text=text, org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[100, 100, 100], lineType=cv2.LINE_AA, thickness=2)


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
out = cv2.VideoWriter('output/cctf_video_out.avi', fourcc, fps, (w, h))

_, prev = cap.read()
prev = cv2.resize(prev, (w, h))
prev1_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
prev2_gray = prev1_gray.copy()
prev3_gray = prev1_gray.copy()
prev4_gray = prev1_gray.copy()
prev5_gray = prev1_gray.copy()
prev6_gray = prev1_gray.copy()


s = prev1_gray.shape
colors1 = (getColor(s, 255, 0, 0))
colors2 = (getColor(s, 0, 126, 0))
colors3 = (getColor(s, 255, 126, 0))
colors4 = (getColor(s, -255, 126, 0))
colors5 = (getColor(s, -75, -75, 255))
colors6= (getColor(s, 255, 0, 255))

def getCctf(g0, g1, g2, g3, g4, g5, g6, doAlign=False):
    ''' g for gray'''
    # doAlign = True
    # im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    #
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
    # this works for rainbow!
    s = g0.shape
    # diff0_1c = changeColor(diff0_1, getColor(s, 255, 0, 0))
    # diff1_2c = changeColor(diff1_2, getColor(s, 0, 126, 0))
    # diff2_3c = changeColor(diff2_3, getColor(s, 255, 126, 0))
    # diff3_4c = changeColor(diff3_4, getColor(s, -255, 126, 0))
    # diff4_5c = changeColor(diff4_5, getColor(s, 255, 0, 255))
    # this works for rainbow!
    diff0_1c = changeColor(diff0_1, colors1)
    diff1_2c = changeColor(diff1_2, colors2)
    diff2_3c = changeColor(diff2_3, colors3)
    diff3_4c = changeColor(diff3_4, colors4)
    diff4_5c = changeColor(diff4_5, colors5)
    diff5_6c = changeColor(diff5_6, colors6)
    #
    cctf = diff0_1c + diff1_2c + diff2_3c + diff3_4c + diff4_5c + diff5_6c
    cctf = cctf * 4
    cctf = np.minimum(cctf, getColor(s, 255, 255, 255))
    cctf = np.maximum(cctf, getColor(s, 0, 0, 0))
    cctf = cctf.astype('uint8')
    return cctf



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
        cctfUnaligned = getCctf(frame, prev1_gray, prev2_gray, prev3_gray, prev4_gray, prev5_gray, prev6_gray, False)
        #
        # writeTextTopLeft(cctfAligned, 'aligned')
        # writeTextTopLeft(cctfUnaligned, 'unaligned')
        # frame_out = cv2.hconcat([frame, prev1_gray, prev2_gray, prev3_gray])
        # frame_out = cv2.hconcat([frame, prev1_gray, prev2_gray, prev3_gray])
        # frame_out = cctf
        frame_out = cctfUnaligned
        # frame_out = cv2.vconcat([cctfUnaligned, frame_color])
        #
        # cv2.imshow("asdf", frame_out)
        # cv2.waitKey(10)
        out.write(frame_out)  # https://stackoverflow.com/a/50076149/8870055
    prev6_gray = prev5_gray#.copy()
    prev5_gray = prev4_gray#.copy()
    prev4_gray = prev3_gray#.copy()
    prev3_gray = prev2_gray#.copy()
    prev2_gray = prev1_gray#.copy()
    prev1_gray = frame#.copy()

# Release video
cap.release()
out.release()
# Close windows
cv2.destroyAllWindows()
#
# # TODO print number on frame screens to compare.
# # or just print then side by side like satya did
