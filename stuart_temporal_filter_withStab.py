# stabilizer is eating the first two frames :(

# project proposal!  make a good deep learning image stabilizer.  go around recording w/ two cameras w/ synched shutters
#image stab can't fix zoom!!!!!  undo zoom effects to focus on object

import os

import cv2

from vidstab import VidStab, download_ostrich_video

# https://github.com/AdamSpannbauer/python_video_stab
# Download test video to stabilize
if not os.path.isfile("ostrich.mp4"):
    download_ostrich_video("ostrich.mp4")

# Initialize object tracker, stabilizer, and video reader
object_tracker = cv2.TrackerCSRT_create()
stabilizer = VidStab()
# vidcap = cv2.VideoCapture("/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/19sec.mov")
vidcap = cv2.VideoCapture("/Users/stuartrobinson/repos/computervision/andre_aigassi/images/tennis_video/forehand.mp4")

# Get frame count
n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

# Get width and height of video stream
w = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
h = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
fps = vidcap.get(cv2.CAP_PROP_FPS)

print(h, w, fps)

# fps = int(vidcap.get(cv2.CV_CAP_PROP_FPS))

# Define the codec for output video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# Set up output video
out = cv2.VideoWriter('video_out.avi', fourcc, fps, (w, h))
# out = cv2.VideoWriter('video_out.mp4', fourcc, fps, (w, h))

import numpy as np

frames = []

# filter = np.array([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
# filter = np.array([-.1, -.1, 0, 1.2])
filter = np.array([1])

count = 0

frame1 = []
frame2 = []

sw = 30

while True:
    count += 1
    success, img = vidcap.read()  # success is boolean
    if success:
        # break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (w, h))

    elif count - n_frames > sw:
        break

    # Pass frame to stabilizer even if frame is None
    # stabilized_frame will be an all black frame until iteration 30
    img = stabilizer.stabilize_frame(input_frame=img, smoothing_window=sw)
    # If stabilized_frame is None then there are no frames left to process
    if img is None:
        break

    if count == sw + 2:
        for i in range(len(filter)):
            # print(img)
            # break
            frames.append(img)

    print("nframes, count:", n_frames, count)
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.fromfunction.html

    if count >= sw + 2:
        frames.pop(0)
        frames.append(img)  # frames.pop(0) to remove first element

        stack = np.dstack([frames[0]])

        for i in range(1, len(frames)):
            stack = np.dstack([stack, frames[i]])  # prob better way to do this

        assert stack.shape[2] == len(frames) == len(filter)

        img = np.absolute(np.dot(stack, filter))
        img = img.astype('uint8')

        out.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))  # https://stackoverflow.com/a/50076149/8870055
        cv2.imshow("Before and After", img)
        key = cv2.waitKey(5)

out.release()
vidcap.release()
cv2.destroyAllWindows()
