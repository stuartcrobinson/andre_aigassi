import cv2
import numpy as np

from cctf.main import getDiff
from imgTools import writeTextTopLeft
from satyaStabilization.satyaHomography2018 import alignHomography2018

# Read input video
# cap = cv2.VideoCapture('video.mp4')
# cap = cv2.VideoCapture('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/forehand.mp4')
cap = cv2.VideoCapture('/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/19sec.mov')

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
print(n_frames, w, h, fps, fourcc)

out = cv2.VideoWriter('output/video_out.avi', fourcc, fps, (w, 2 * h))

_, prev = cap.read()
prev = cv2.resize(prev, (w, h))
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)


for i in range(n_frames - 2):
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.resize(frame, (w, h))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    alignedFrame = alignHomography2018(frame, frame_gray, prev_gray)
    alignedFrame_gray = cv2.cvtColor(alignedFrame, cv2.COLOR_BGR2GRAY)
    #
    diff_gray_aligned = getDiff(alignedFrame_gray, prev_gray)
    diff_gray_unaligned = getDiff(frame_gray, prev_gray)
    #
    diff_aligned = cv2.cvtColor(diff_gray_aligned, cv2.COLOR_GRAY2BGR)
    diff_unaligned = cv2.cvtColor(diff_gray_unaligned, cv2.COLOR_GRAY2BGR)
    #
    writeTextTopLeft(diff_aligned, 'aligned')
    writeTextTopLeft(diff_unaligned, 'unaligned')
    frame_out = cv2.vconcat([diff_aligned, diff_unaligned])
    #
    cv2.imshow("asdf", frame_out)
    cv2.waitKey(10)
    out.write(frame_out)  # https://stackoverflow.com/a/50076149/8870055
    prev_gray = frame_gray

# Release video
cap.release()
out.release()
# Close windows
cv2.destroyAllWindows()

# TODO print number on frame screens to compare.
# or just print then side by side like satya did
