import cv2
import numpy as np

from satyaHomography2018 import alignHomography2018

# Read input video
cap = cv2.VideoCapture('video.mp4')
# cap = cv2.VideoCapture('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/forehand.mp4')

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

out = cv2.VideoWriter('video_out.avi', fourcc, fps, (w, h))

_, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

startFrameI = 1
lastFrameI = 10000

startFrame = []

for i in range(n_frames - 2):
    success, frame = cap.read()
    # if i == startFrameI - 1:
    #     startFrame = frame
    #     prev = frame
    #     prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    if not success:
        break
    # if i < startFrameI:
    #     continue
    # if i > lastFrameI:
    #     break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    alignedFrame, h = alignHomography2018(frame, prev)
    # alignedFrame = frame
    alignedFrame_gray = cv2.cvtColor(alignedFrame, cv2.COLOR_BGR2GRAY)
    #
    # diff_gray = np.absolute(frame_gray - prev_gray) # why looks like shit

    stack = np.dstack([prev_gray, alignedFrame_gray])
    # stack = np.dstack([prev_gray, frame_gray])
    diff_gray = np.absolute(np.dot(stack, [-1, 1])).astype('uint8')
    # diff = diff.astype('uint8')

    # diff = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    diff = cv2.cvtColor(diff_gray, cv2.COLOR_GRAY2BGR)
    #
    cv2.imshow("Before and After", diff)
    cv2.waitKey(10)
    out.write(diff)  # https://stackoverflow.com/a/50076149/8870055
    # out.write(frame)  # https://stackoverflow.com/a/50076149/8870055
    prev_gray = frame_gray
    prev = frame

# Release video
cap.release()
out.release()
# Close windows
cv2.destroyAllWindows()


TODO print number on frame screens to compare.
or just print then side by side like satya did